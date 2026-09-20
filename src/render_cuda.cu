#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cmath>
#include <utility>
#include "render_cuda.h"
#include "bla.h"
#include "shade.cuh"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                         __LINE__, cudaGetErrorString(err__));                    \
            return false;                                                         \
        }                                                                         \
    } while (0)

// Per-pixel iteration state carried between slices.
struct PixelState {
    double dzr, dzi;  // dz = z - Z_m
    double dr, di;    // dz/dc * scale, for the distance estimate
    int m;            // reference index
    int n;            // pixel iteration
    int status;       // 0 active, 1 escaped, 2 inside
    int pad;
};

namespace {

// GPU time per slice launch. Short enough that the composite on the other
// stream and any pending view change wait at most this long.
constexpr unsigned long long kSliceBudgetNs = 20ull * 1000000ull;

__device__ __forceinline__ FieldSample emptySample() { return FieldSample{}; }  // gen 0

// New pass: reset iteration state only. Field samples from earlier
// generations stay until overwritten.
__global__ void resetState(PixelState* __restrict__ st, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    st[i] = PixelState{};
}

__global__ void clearField(FieldSample* __restrict__ field, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    field[i] = emptySample();
}

// Reset only pixels that have not finished (their dz referred to an old
// reference orbit).
__global__ void resetPendingState(PixelState* __restrict__ st, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (st[i].status == 0) st[i] = PixelState{};
}

// dst(x, y) = src(x + dx, y + dy), pending where the source is off-image.
__global__ void shiftKernel(const PixelState* __restrict__ srcS,
                            const FieldSample* __restrict__ srcF, PixelState* __restrict__ dstS,
                            FieldSample* __restrict__ dstF, int w, int h, int dx, int dy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int sx = x + dx, sy = y + dy;
    const size_t di = (size_t)y * w + x;
    if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
        const size_t si = (size_t)sy * w + sx;
        dstS[di] = srcS[si];
        dstF[di] = srcF[si];
    } else {
        dstS[di] = PixelState{};
        dstF[di] = emptySample();
    }
}

__device__ __forceinline__ void writeSample(FieldSample* __restrict__ field, size_t idx,
                                            const PixelState& st, double zr, double zi,
                                            double scale, bool escaped, float gen) {
    FieldSample s{};
    if (!escaped) {
        s.iter = -1.f;
    } else {
        const double mag2 = zr * zr + zi * zi;
        const double logMag = 0.5 * log(mag2);
        s.iter = (float)(st.n + 1.0 - log2(logMag / 0.6931471805599453));
        const double dmag2 = st.dr * st.dr + st.di * st.di;
        const double dmag = sqrt(dmag2);
        // Derivative was scaled by pixel size; undo it so de is in complex units.
        s.de = dmag > 0.0 ? (float)(sqrt(mag2) * logMag / dmag * scale) : 0.f;
        s.angle = (float)atan2(zi, zr);
        // Milnor normal u = z / dz, normalised.
        if (dmag2 > 0.0) {
            const double ur = (zr * st.dr + zi * st.di) / dmag2;
            const double ui = (zi * st.dr - zr * st.di) / dmag2;
            const double um = sqrt(ur * ur + ui * ui);
            if (um > 0.0) {
                s.nx = (float)(ur / um);
                s.ny = (float)(ui / um);
            }
        }
    }
    s.gen = gen;
    field[idx] = s;
}

__device__ __forceinline__ unsigned long long globalTimerNs() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Stamps the GPU clock right before the slice so the deadline is relative
// to when the GPU actually started it, not when the host queued it.
__global__ void stampTimer(unsigned long long* __restrict__ out) { *out = globalTimerNs(); }

// Find the longest BLA node starting at reference index m that is valid
// for |dz|^2 = dzmag2. Returns nullptr if none.
__device__ __forceinline__ const BlaNode* findBla(const DeviceBla& bla, int m, double dzmag2,
                                                  int refLast) {
    const int j0 = m - 1;
    if (j0 < 0 || j0 >= bla.steps) return nullptr;
    int k = j0 == 0 ? bla.levels - 1 : min(bla.levels - 1, __ffs(j0) - 1);
    for (; k >= 0; --k) {
        const BlaNode* n = bla.nodes + bla.levelOffset[k] + (j0 >> k);
        if (dzmag2 < n->r2 && m + n->l <= refLast) return n;
    }
    return nullptr;
}

// Perturbation iteration in double precision with Zhuoran's rebasing and
// BLA skipping, advanced by at most sliceIters iterations per launch and
// never past the GPU-clock deadline.
//   dz_{n+1} = 2 Z_m dz_n + dz_n^2 + dc
//   rebase when |Z_m + dz| < |dz|: dz = Z_m + dz, m = 0
// Field layout: fw x fh with the view (w x h) at offset (mx, my). Pixel
// (x, y) in field coordinates sits at view offset (x - mx, y - my).
__global__ void iterateSlice(PixelState* __restrict__ state, FieldSample* __restrict__ field,
                             int fw, int fh, int mx, int my, int w, int h, int stride,
                             double scale, double refOffX, double refOffY, int maxIter,
                             int sliceIters, const unsigned long long* __restrict__ startNs,
                             unsigned long long budgetNs, DeviceReference ref, DeviceBla bla,
                             float gen, int* __restrict__ activeCount) {
    const unsigned long long deadlineNs = *startNs + budgetNs;
    int x, y;
    bool inBounds;
    if (stride == 1) {
        // Full resolution only inside the view; the margin stays coarse.
        x = mx + blockIdx.x * blockDim.x + threadIdx.x;
        y = my + blockIdx.y * blockDim.y + threadIdx.y;
        inBounds = x < mx + w && y < my + h;
    } else {
        x = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
        y = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
        inBounds = x < fw && y < fh;
    }
    const size_t idx = inBounds ? (size_t)y * fw + x : 0;
    PixelState st{};
    if (inBounds) st = state[idx];
    bool active = inBounds && st.status == 0;

    if (active) {
        const double dcr = refOffX + ((double)(x - mx) - 0.5 * w) * scale;
        const double dci = refOffY - ((double)(y - my) - 0.5 * h) * scale;
        double dzr = st.dzr, dzi = st.dzi, dr = st.dr, di = st.di;
        int m = st.m, n = st.n;
        double zr = 0.0, zi = 0.0;
        const double bailout = 65536.0;
        const int refLast = ref.length - 1;
        const int stop = min(maxIter, n + sliceIters);
        bool escaped = false;
        double dzmag2 = dzr * dzr + dzi * dzi;
        int budgetCheck = 0;

        while (n < stop) {
            if ((++budgetCheck & 63) == 0 && globalTimerNs() > deadlineNs) break;
            if (bla.enabled) {
                const BlaNode* nd = findBla(bla, m, dzmag2, refLast);
                if (nd) {
                    // dz' = A dz + B dc ; d' = A d + B scale
                    const double ndzr = nd->ar * dzr - nd->ai * dzi + nd->br * dcr - nd->bi * dci;
                    const double ndzi = nd->ar * dzi + nd->ai * dzr + nd->br * dci + nd->bi * dcr;
                    const double ndr = nd->ar * dr - nd->ai * di + nd->br * scale;
                    const double ndi = nd->ar * di + nd->ai * dr + nd->bi * scale;
                    dzr = ndzr;
                    dzi = ndzi;
                    dr = ndr;
                    di = ndi;
                    m += nd->l;
                    n += nd->l;
                    zr = ref.zr[m] + dzr;
                    zi = ref.zi[m] + dzi;
                    const double zmag2 = zr * zr + zi * zi;
                    if (zmag2 > bailout) {
                        escaped = true;
                        break;
                    }
                    dzmag2 = dzr * dzr + dzi * dzi;
                    if (zmag2 < dzmag2 || m >= refLast) {
                        dzr = zr;
                        dzi = zi;
                        dzmag2 = zmag2;
                        m = 0;
                    }
                    continue;
                }
            }
            const double Zr = ref.zr[m];
            const double Zi = ref.zi[m];
            const double fzr = Zr + dzr, fzi = Zi + dzi;
            const double ndr = 2.0 * (fzr * dr - fzi * di) + scale;
            const double ndi = 2.0 * (fzr * di + fzi * dr);
            dr = ndr;
            di = ndi;
            const double ar = 2.0 * Zr + dzr, ai = 2.0 * Zi + dzi;
            const double ndzr = ar * dzr - ai * dzi + dcr;
            const double ndzi = ar * dzi + ai * dzr + dci;
            dzr = ndzr;
            dzi = ndzi;
            ++m;
            ++n;
            zr = ref.zr[m] + dzr;
            zi = ref.zi[m] + dzi;
            const double zmag2 = zr * zr + zi * zi;
            if (zmag2 > bailout) {
                escaped = true;
                break;
            }
            dzmag2 = dzr * dzr + dzi * dzi;
            if (zmag2 < dzmag2 || m >= refLast) {
                dzr = zr;
                dzi = zi;
                dzmag2 = zmag2;
                m = 0;
            }
        }
        st.dzr = dzr;
        st.dzi = dzi;
        st.dr = dr;
        st.di = di;
        st.m = m;
        st.n = n;
        if (escaped) st.status = 1;
        else if (n >= maxIter) st.status = 2;
        const bool pending = st.status == 0;
        // In-progress pixels keep whatever older sample sits there.
        if (!pending) writeSample(field, idx, st, zr, zi, scale, escaped, gen);
        state[idx] = st;
        active = pending;
    }

    // Warp-aggregated count of pixels still active.
    const unsigned mask = __ballot_sync(0xffffffffu, active);
    if ((threadIdx.x & 31) == 0 && mask) atomicAdd(activeCount, __popc(mask));
}

// Best sample of generation g at field pixel (x, y): the pixel itself, or
// the nearest coarse-level anchor (stride 2, 4, 8) if the pixel itself was
// not produced by that generation. level receives 1, 2, 4, 8 (0 = none).
__device__ __forceinline__ FieldSample fetchGen(const FieldSample* __restrict__ field, int fw,
                                                int x, int y, float g, int& level) {
    FieldSample s = field[(size_t)y * fw + x];
    if (s.gen == g) {
        level = 1;
        return s;
    }
    for (int k = 2; k <= 8; k *= 2) {
        const int bx = x - (x % k), by = y - (y % k);
        const FieldSample a = field[(size_t)by * fw + bx];
        if (a.gen == g) {
            level = k;
            return a;
        }
    }
    level = 0;
    return s;
}

// Shade one field sample (with same-generation neighbour gradient for lines).
__device__ __forceinline__ float3 shadeField(const FieldSample* __restrict__ field, int fw, int fh,
                                             int xi, int yi, const FieldSample& s,
                                             const ShadeParams& p, float timeSec,
                                             const GenMap& gm) {
    IterGradient g;
    if (p.lines && xi + 1 < fw && yi + 1 < fh && s.iter >= 0.f) {
        const FieldSample sx = field[(size_t)yi * fw + xi + 1];
        const FieldSample sy = field[(size_t)(yi + 1) * fw + xi];
        if (sx.gen == s.gen && sy.gen == s.gen && sx.iter >= 0.f && sy.iter >= 0.f) {
            g.dx = (sx.iter - s.iter) * (float)gm.ratio;
            g.dy = (sy.iter - s.iter) * (float)gm.ratio;
            g.valid = true;
        }
    }
    return shadeSample(s, p, timeSec, gm.pixelScale, g);
}

// One subsample: search the generations for the best data at this point.
// Returns false if no generation has anything here.
__device__ __forceinline__ bool sampleBest(const FieldSample* __restrict__ field, int fw, int fh,
                                           int mx, int my, double px, double py,
                                           const CompositeMap& map, const ShadeParams& p,
                                           float timeSec, float3& col) {
    // Newest generation: exact pixel or coarse anchor.
    float bestDetail = 0.f;
    FieldSample best{};
    int bestX = 0, bestY = 0, bestGen = -1;
    for (int k = 0; k < map.genCount; ++k) {
        const GenMap& gm = map.gens[k];
        const double u = gm.ox + px * gm.ratio + mx, v = gm.oy + py * gm.ratio + my;
        const int xi = (int)floor(u + 0.5), yi = (int)floor(v + 0.5);
        if (xi < 0 || xi >= fw || yi < 0 || yi >= fh) continue;
        int level = 0;
        const FieldSample s = fetchGen(field, fw, xi, yi, gm.gen, level);
        if (level == 0) continue;
        const float detail = (float)gm.ratio / (float)level;  // field px per display px
        if (detail > bestDetail) {
            bestDetail = detail;
            best = s;
            bestX = xi;
            bestY = yi;
            bestGen = k;
        }
        // An exact hit in this generation beats anything older unless the
        // older one is genuinely finer (zoomed out). Stop early when the
        // remaining generations cannot beat it.
        if (level == 1 && k + 1 < map.genCount && map.gens[k + 1].ratio <= gm.ratio) break;
    }
    if (bestGen < 0) {
        // Nothing covers this point: extend the nearest field edge of the
        // first generation with data there. Streaky, but only for the frame
        // or two before the new pass's coarse level lands.
        for (int k = 0; k < map.genCount; ++k) {
            const GenMap& gm = map.gens[k];
            const double u = gm.ox + px * gm.ratio + mx, v = gm.oy + py * gm.ratio + my;
            const int xi = min(max((int)floor(u + 0.5), 0), fw - 1);
            const int yi = min(max((int)floor(v + 0.5), 0), fh - 1);
            int level = 0;
            const FieldSample sm = fetchGen(field, fw, xi, yi, gm.gen, level);
            if (level == 0) continue;
            col = shadeField(field, fw, fh, xi, yi, sm, p, timeSec, gm);
            return true;
        }
        return false;
    }
    col = shadeField(field, fw, fh, bestX, bestY, best, p, timeSec, map.gens[bestGen]);
    return true;
}

// Display composite: ss x ss subsamples per display pixel, each taken from
// the generation with the most detail at that point.
__global__ void compositeKernel(const FieldSample* __restrict__ field, int fw, int fh, int mx,
                                int my, uint32_t* __restrict__ out, int w, int h, ShadeParams p,
                                float timeSec, CompositeMap map) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t idx = (size_t)y * w + x;
    const double px = (double)x - 0.5 * w, py = (double)y - 0.5 * h;
    const int ss = map.ss < 1 ? 1 : map.ss;
    float3 acc = make_float3(0.f, 0.f, 0.f);
    int have = 0;
    for (int j = 0; j < ss; ++j) {
        for (int i = 0; i < ss; ++i) {
            const double sx = px + (i + 0.5) / ss - 0.5, sy = py + (j + 0.5) / ss - 0.5;
            float3 c;
            if (sampleBest(field, fw, fh, mx, my, sx, sy, map, p, timeSec, c)) {
                acc.x += c.x;
                acc.y += c.y;
                acc.z += c.z;
                ++have;
            }
        }
    }
    if (have > 0) {
        const float inv = 1.f / have;
        out[idx] = packSRGB8(make_float3(acc.x * inv, acc.y * inv, acc.z * inv));
    } else {
        out[idx] = 0xFF000000u;
    }
}

}  // namespace

CudaRenderer::~CudaRenderer() {
    unregisterPbo();
    freeField();
    freeReference();
    freeBla();
    if (evStart_) cudaEventDestroy((cudaEvent_t)evStart_);
    if (evStop_) cudaEventDestroy((cudaEvent_t)evStop_);
    if (evSlice_) cudaEventDestroy((cudaEvent_t)evSlice_);
    if (evShadeA_) cudaEventDestroy((cudaEvent_t)evShadeA_);
    if (evShadeB_) cudaEventDestroy((cudaEvent_t)evShadeB_);
    if (activeCount_) cudaFree(activeCount_);
    if (sliceStart_ns_) cudaFree(sliceStart_ns_);
    if (activeCountHost_) cudaFreeHost(activeCountHost_);
    if (stream_) cudaStreamDestroy((cudaStream_t)stream_);
    if (dispStream_) cudaStreamDestroy((cudaStream_t)dispStream_);
}

bool CudaRenderer::init() {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::snprintf(deviceName_, sizeof deviceName_, "%s (SM %d.%d, %d SMs)", prop.name,
                  prop.major, prop.minor, prop.multiProcessorCount);
    cudaEvent_t a, b, es, sa, sb;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventCreateWithFlags(&es, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&sa));
    CUDA_CHECK(cudaEventCreate(&sb));
    evStart_ = a;
    evStop_ = b;
    evSlice_ = es;
    evShadeA_ = sa;
    evShadeB_ = sb;
    cudaStream_t st, ds;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ds, cudaStreamNonBlocking));
    stream_ = st;
    dispStream_ = ds;
    CUDA_CHECK(cudaMalloc(&activeCount_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sliceStart_ns_, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocHost(&activeCountHost_, sizeof(int)));
    *activeCountHost_ = 0;
    return true;
}

void CudaRenderer::syncAll() {
    if (stream_) cudaStreamSynchronize((cudaStream_t)stream_);
    if (dispStream_) cudaStreamSynchronize((cudaStream_t)dispStream_);
}

void CudaRenderer::unregisterPbo() {
    if (pboResource_) {
        syncAll();
        cudaGraphicsUnregisterResource(pboResource_);
        pboResource_ = nullptr;
    }
}

void CudaRenderer::freeField() {
    syncAll();
    if (field_) cudaFree(field_);
    if (state_) cudaFree(state_);
    if (fieldAlt_) cudaFree(fieldAlt_);
    if (stateAlt_) cudaFree(stateAlt_);
    field_ = fieldAlt_ = nullptr;
    state_ = stateAlt_ = nullptr;
    iterDone_ = true;
    sliceInFlight_ = false;
}

void CudaRenderer::freeReference() {
    syncAll();
    if (refZr_) cudaFree(refZr_);
    if (refZi_) cudaFree(refZi_);
    refZr_ = refZi_ = nullptr;
    refCapacity_ = 0;
    ref_ = DeviceReference{};
}

void CudaRenderer::freeBla() {
    syncAll();
    if (blaNodes_) cudaFree(blaNodes_);
    if (blaOffsets_) cudaFree(blaOffsets_);
    blaNodes_ = nullptr;
    blaOffsets_ = nullptr;
    blaNodeCapacity_ = blaLevelCapacity_ = 0;
    bla_ = DeviceBla{};
}

bool CudaRenderer::uploadBla(const BlaNode* nodes, int count, const int* levelOffset, int levels,
                             int steps) {
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream_));
    if (count > blaNodeCapacity_ || levels > blaLevelCapacity_) {
        freeBla();
        blaNodeCapacity_ = count + count / 4 + 1024;
        blaLevelCapacity_ = levels + 8;
        CUDA_CHECK(cudaMalloc(&blaNodes_, sizeof(BlaNode) * (size_t)blaNodeCapacity_));
        CUDA_CHECK(cudaMalloc(&blaOffsets_, sizeof(int) * (size_t)blaLevelCapacity_));
    }
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(blaNodes_, nodes, sizeof(BlaNode) * (size_t)count,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(blaOffsets_, levelOffset, sizeof(int) * (size_t)levels,
                              cudaMemcpyHostToDevice));
    }
    bla_.nodes = blaNodes_;
    bla_.levelOffset = blaOffsets_;
    bla_.levels = levels;
    bla_.steps = steps;
    return true;
}

bool CudaRenderer::uploadReference(const double* zr, const double* zi, int length, bool escaped) {
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream_));  // no slice may read the old orbit
    if (length > refCapacity_) {
        freeReference();
        refCapacity_ = length + length / 4 + 1024;
        CUDA_CHECK(cudaMalloc(&refZr_, sizeof(double) * (size_t)refCapacity_));
        CUDA_CHECK(cudaMalloc(&refZi_, sizeof(double) * (size_t)refCapacity_));
    }
    CUDA_CHECK(cudaMemcpy(refZr_, zr, sizeof(double) * (size_t)length, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(refZi_, zi, sizeof(double) * (size_t)length, cudaMemcpyHostToDevice));
    ref_.zr = refZr_;
    ref_.zi = refZi_;
    ref_.length = length;
    ref_.escaped = escaped;
    return true;
}

bool CudaRenderer::bindPixelBuffer(unsigned glPbo, int width, int height, int ss) {
    unregisterPbo();
    freeField();
    width_ = width;
    height_ = height;
    ss_ = ss < 1 ? 1 : ss;
    if (!glPbo || width <= 0 || height <= 0) return true;
    viewW_ = width * ss_;
    viewH_ = height * ss_;
    marginX_ = (viewW_ + 7) / 8;
    marginY_ = (viewH_ + 7) / 8;
    fieldW_ = viewW_ + 2 * marginX_;
    fieldH_ = viewH_ + 2 * marginY_;
    const size_t n = (size_t)width * height;
    const size_t fn = (size_t)fieldW_ * fieldH_;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&pboResource_, glPbo,
                                            cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK(cudaMalloc(&field_, sizeof(FieldSample) * fn));
    CUDA_CHECK(cudaMalloc(&state_, sizeof(PixelState) * fn));
    CUDA_CHECK(cudaMalloc(&fieldAlt_, sizeof(FieldSample) * fn));
    CUDA_CHECK(cudaMalloc(&stateAlt_, sizeof(PixelState) * fn));
    (void)n;
    clearField<<<(int)((fn + 255) / 256), 256>>>(field_, (int)fn);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

void CudaRenderer::resetPass(const ViewParams& view) {
    iterView_ = view;
    bla_.enabled = view.useBla ? 1 : 0;
    sliceStart_ = 0;
    sliceIters_ = view.maxIter;  // the GPU-clock deadline bounds each launch
    sliceInFlight_ = false;
    stride_ = firstStride_;
    completedStride_ = 0;
    iterDone_ = false;
    passMs_ = 0.f;
}

bool CudaRenderer::beginIterate(const ViewParams& view, float gen) {
    if (!field_ || !state_ || view.width != viewW_ || view.height != viewH_) return false;
    if (ref_.length < 2) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    CUDA_CHECK(cudaStreamSynchronize(stream));  // cancel the slice in flight
    resetPass(view);
    gen_ = gen;
    const int count = fieldW_ * fieldH_;
    resetState<<<(count + 255) / 256, 256, 0, stream>>>(state_, count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool CudaRenderer::shiftAndResume(const ViewParams& view, int dx, int dy) {
    if (!field_ || !state_ || view.width != viewW_ || view.height != viewH_) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    syncAll();  // the composite may be reading the buffers we are about to swap
    if (dx != 0 || dy != 0) {
        const dim3 block(32, 8);
        const dim3 grid((fieldW_ + block.x - 1) / block.x, (fieldH_ + block.y - 1) / block.y);
        shiftKernel<<<grid, block, 0, stream>>>(state_, field_, stateAlt_, fieldAlt_, fieldW_,
                                                fieldH_, dx, dy);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::swap(state_, stateAlt_);
        std::swap(field_, fieldAlt_);
    }
    const float keepPass = passMs_;
    resetPass(view);
    passMs_ = keepPass;
    return true;
}

bool CudaRenderer::restartPending(const ViewParams& view) {
    if (!field_ || !state_) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int count = fieldW_ * fieldH_;
    resetPendingState<<<(count + 255) / 256, 256, 0, stream>>>(state_, count);
    CUDA_CHECK(cudaGetLastError());
    const float keepPass = passMs_;
    resetPass(view);
    passMs_ = keepPass;
    return true;
}

bool CudaRenderer::iterateBusy() {
    if (!sliceInFlight_) return false;
    const cudaError_t q = cudaEventQuery((cudaEvent_t)evSlice_);
    if (q == cudaErrorNotReady) return true;
    // Slice finished: collect timing and the active count.
    sliceInFlight_ = false;
    sliceFinished_ = true;
    cudaEventElapsedTime(&sliceMs_, (cudaEvent_t)evStart_, (cudaEvent_t)evStop_);
    passMs_ += sliceMs_;
    iterateMs_ = passMs_;
    if (*activeCountHost_ == 0) {
        completedStride_ = stride_;
        if (stride_ <= 1) {
            iterDone_ = true;
        } else {
            stride_ /= 2;
            sliceStart_ = 0;
        }
    }
    return false;
}

bool CudaRenderer::stepIterate() {
    if (iterDone_ || sliceInFlight_) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    const dim3 block(32, 8);
    int sw, sh;
    if (stride_ == 1) {
        sw = viewW_;
        sh = viewH_;
    } else {
        sw = (fieldW_ + stride_ - 1) / stride_;
        sh = (fieldH_ + stride_ - 1) / stride_;
    }
    const dim3 grid((sw + block.x - 1) / block.x, (sh + block.y - 1) / block.y);
    CUDA_CHECK(cudaMemsetAsync(activeCount_, 0, sizeof(int), stream));
    cudaEventRecord((cudaEvent_t)evStart_, stream);
    stampTimer<<<1, 1, 0, stream>>>(sliceStart_ns_);
    iterateSlice<<<grid, block, 0, stream>>>(state_, field_, fieldW_, fieldH_, marginX_, marginY_,
                                             viewW_, viewH_, stride_, iterView_.scale,
                                             iterView_.refOffX, iterView_.refOffY,
                                             iterView_.maxIter, sliceIters_, sliceStart_ns_,
                                             kSliceBudgetNs, ref_, bla_, gen_, activeCount_);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord((cudaEvent_t)evStop_, stream);
    CUDA_CHECK(cudaMemcpyAsync(activeCountHost_, activeCount_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    cudaEventRecord((cudaEvent_t)evSlice_, stream);
    sliceStart_ = iterView_.maxIter;
    sliceInFlight_ = true;
    return true;
}

bool CudaRenderer::composite(const ShadeParams& params, float timeSec, const CompositeMap& map) {
    if (!pboResource_ || !field_) return false;
    cudaStream_t stream = (cudaStream_t)dispStream_;
    CUDA_CHECK(cudaGraphicsMapResources(1, &pboResource_, stream));
    uint32_t* devPtr = nullptr;
    size_t bytes = 0;
    cudaError_t err = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &bytes, pboResource_);
    if (err != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &pboResource_, stream);
        std::fprintf(stderr, "map failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    // Timing of the previous composite, harvested without blocking.
    if (shadePending_ && cudaEventQuery((cudaEvent_t)evShadeB_) == cudaSuccess) {
        cudaEventElapsedTime(&shadeMs_, (cudaEvent_t)evShadeA_, (cudaEvent_t)evShadeB_);
        shadePending_ = false;
    }
    if (!shadePending_) cudaEventRecord((cudaEvent_t)evShadeA_, stream);
    compositeKernel<<<grid, block, 0, stream>>>(field_, fieldW_, fieldH_, marginX_, marginY_,
                                                devPtr, width_, height_, params, timeSec, map);
    err = cudaGetLastError();
    if (!shadePending_) {
        cudaEventRecord((cudaEvent_t)evShadeB_, stream);
        shadePending_ = true;
    }
    // Unmap is stream-ordered; later GL calls wait for it without a host sync.
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &pboResource_, stream));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "composite kernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}
