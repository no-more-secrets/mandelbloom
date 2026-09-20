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

__device__ __forceinline__ FieldSample pendingSample() {
    FieldSample s{};
    s.flags = 1.f;
    return s;
}

__global__ void resetState(PixelState* __restrict__ st, FieldSample* __restrict__ field,
                           int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    st[i] = PixelState{};
    field[i] = pendingSample();
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
        dstF[di] = pendingSample();
    }
}

__device__ __forceinline__ void writeSample(FieldSample* __restrict__ field, size_t idx,
                                            const PixelState& st, double zr, double zi,
                                            double scale, bool escaped, bool pending) {
    FieldSample s{};
    if (!escaped) {
        s.iter = pending ? (float)st.n : -1.f;
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
    s.flags = pending ? 1.f : 0.f;
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
                             int* __restrict__ activeCount) {
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
        writeSample(field, idx, st, zr, zi, scale, escaped, pending);
        state[idx] = st;
        active = pending;
    }

    // Warp-aggregated count of pixels still active.
    const unsigned mask = __ballot_sync(0xffffffffu, active);
    if ((threadIdx.x & 31) == 0 && mask) atomicAdd(activeCount, __popc(mask));
}

// The pixel's own sample, or the nearest finished coarse-level anchor
// (stride 2, 4, 8) if the pixel itself is still pending.
__device__ __forceinline__ FieldSample fetchFilled(const FieldSample* __restrict__ field, int w,
                                                   int x, int y) {
    FieldSample s = field[(size_t)y * w + x];
    if (s.flags < 0.5f) return s;
    for (int k = 2; k <= 8; k *= 2) {
        const int bx = x - (x % k), by = y - (y % k);
        const FieldSample a = field[(size_t)by * w + bx];
        if (a.flags < 0.5f) return a;
    }
    return s;
}

__device__ __forceinline__ float4 unpack(uint32_t c) {
    return make_float4((float)(c & 255u), (float)((c >> 8) & 255u), (float)((c >> 16) & 255u), 1.f);
}

__device__ __forceinline__ uint32_t bilinear(const uint32_t* __restrict__ img, int w, int h,
                                             double u, double v) {
    const int x0 = (int)u, y0 = (int)v;
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = (float)(u - x0), fy = (float)(v - y0);
    const float4 a = unpack(img[(size_t)y0 * w + x0]), b = unpack(img[(size_t)y0 * w + x1]);
    const float4 c = unpack(img[(size_t)y1 * w + x0]), d = unpack(img[(size_t)y1 * w + x1]);
    const float r = (a.x * (1 - fx) + b.x * fx) * (1 - fy) + (c.x * (1 - fx) + d.x * fx) * fy;
    const float g = (a.y * (1 - fx) + b.y * fx) * (1 - fy) + (c.y * (1 - fx) + d.y * fx) * fy;
    const float bl = (a.z * (1 - fx) + b.z * fx) * (1 - fy) + (c.z * (1 - fx) + d.z * fx) * fy;
    return (uint32_t)(r + 0.5f) | ((uint32_t)(g + 0.5f) << 8) | ((uint32_t)(bl + 0.5f) << 16) |
           0xFF000000u;
}

// Display composite: per pixel pick the running pass's field or the last
// finished frame, whichever has more detail for this display pixel.
__global__ void compositeKernel(const FieldSample* __restrict__ field, int fw, int fh, int mx,
                                int my, const uint32_t* __restrict__ old,
                                uint32_t* __restrict__ out, int w, int h, ShadeParams p,
                                float timeSec, CompositeMap map) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t idx = (size_t)y * w + x;
    const double px = (double)x - 0.5 * w, py = (double)y - 0.5 * h;

    // Running pass (map is in view pixels; the field adds its margin).
    const double u = map.nox + px * map.ratioN + mx, v = map.noy + py * map.ratioN + my;
    const int xi = (int)floor(u + 0.5), yi = (int)floor(v + 0.5);
    FieldSample s{};
    bool haveNew = false;
    if (xi >= 0 && xi < fw && yi >= 0 && yi < fh) {
        s = fetchFilled(field, fw, xi, yi);
        haveNew = s.flags < 0.5f;
    }
    // Last finished frame.
    const double ou = map.oox + px * map.ratioO, ov = map.ooy + py * map.ratioO;
    const bool haveOld = map.oldDetail > 0.f && ou >= 0.0 && ov >= 0.0 &&
                         ou <= (double)(w - 1) && ov <= (double)(h - 1);

    const bool useNew = haveNew && (!haveOld || map.newDetail >= map.oldDetail);
    if (useNew || (haveNew && !haveOld)) {
        IterGradient g;
        if (p.lines && xi + 1 < fw && yi + 1 < fh && s.iter >= 0.f) {
            // Forward differences in field pixels, scaled to display pixels.
            const FieldSample sx = fetchFilled(field, fw, xi + 1, yi);
            const FieldSample sy = fetchFilled(field, fw, xi, yi + 1);
            if (sx.flags < 0.5f && sy.flags < 0.5f && sx.iter >= 0.f && sy.iter >= 0.f) {
                g.dx = (sx.iter - s.iter) * (float)map.ratioN;
                g.dy = (sy.iter - s.iter) * (float)map.ratioN;
                g.valid = true;
            }
        }
        out[idx] = packSRGB8(shadeSample(s, p, timeSec, map.pixelScaleN, g));
    } else if (haveOld) {
        out[idx] = bilinear(old, w, h, ou, ov);
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
    if (lastImage_) cudaFree(lastImage_);
    field_ = fieldAlt_ = nullptr;
    state_ = stateAlt_ = nullptr;
    lastImage_ = nullptr;
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

bool CudaRenderer::bindPixelBuffer(unsigned glPbo, int width, int height) {
    unregisterPbo();
    freeField();
    width_ = width;
    height_ = height;
    if (!glPbo || width <= 0 || height <= 0) return true;
    marginX_ = (width + 7) / 8;
    marginY_ = (height + 7) / 8;
    fieldW_ = width + 2 * marginX_;
    fieldH_ = height + 2 * marginY_;
    const size_t n = (size_t)width * height;
    const size_t fn = (size_t)fieldW_ * fieldH_;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&pboResource_, glPbo,
                                            cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK(cudaMalloc(&field_, sizeof(FieldSample) * fn));
    CUDA_CHECK(cudaMalloc(&state_, sizeof(PixelState) * fn));
    CUDA_CHECK(cudaMalloc(&fieldAlt_, sizeof(FieldSample) * fn));
    CUDA_CHECK(cudaMalloc(&stateAlt_, sizeof(PixelState) * fn));
    CUDA_CHECK(cudaMalloc(&lastImage_, sizeof(uint32_t) * n));
    CUDA_CHECK(cudaMemset(lastImage_, 0, sizeof(uint32_t) * n));
    return true;
}

void CudaRenderer::resetPass(const ViewParams& view) {
    iterView_ = view;
    bla_.enabled = view.useBla ? 1 : 0;
    sliceStart_ = 0;
    sliceIters_ = 512;  // modest start each pass; the GPU deadline bounds it anyway
    sliceInFlight_ = false;
    stride_ = firstStride_;
    completedStride_ = 0;
    iterDone_ = false;
    passMs_ = 0.f;
}

bool CudaRenderer::beginIterate(const ViewParams& view) {
    if (!field_ || !state_ || view.width != width_ || view.height != height_) return false;
    if (ref_.length < 2) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    CUDA_CHECK(cudaStreamSynchronize(stream));  // cancel the slice in flight
    resetPass(view);
    const int count = fieldW_ * fieldH_;
    resetState<<<(count + 255) / 256, 256, 0, stream>>>(state_, field_, count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool CudaRenderer::shiftAndResume(const ViewParams& view, int dx, int dy) {
    if (!field_ || !state_ || view.width != width_ || view.height != height_) return false;
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
    // Adapt the slice length toward ~25 ms of GPU time.
    if (sliceMs_ > 0.f) {
        float f = 25.f / sliceMs_;
        f = fminf(fmaxf(f, 0.5f), 4.f);
        sliceIters_ = (int)fminf(fmaxf((float)sliceIters_ * f, 16.f), 65536.f);
    }
    return false;
}

bool CudaRenderer::stepIterate() {
    if (iterDone_ || sliceInFlight_) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    const dim3 block(32, 8);
    int sw, sh;
    if (stride_ == 1) {
        sw = width_;
        sh = height_;
    } else {
        sw = (fieldW_ + stride_ - 1) / stride_;
        sh = (fieldH_ + stride_ - 1) / stride_;
    }
    const dim3 grid((sw + block.x - 1) / block.x, (sh + block.y - 1) / block.y);
    CUDA_CHECK(cudaMemsetAsync(activeCount_, 0, sizeof(int), stream));
    cudaEventRecord((cudaEvent_t)evStart_, stream);
    stampTimer<<<1, 1, 0, stream>>>(sliceStart_ns_);
    iterateSlice<<<grid, block, 0, stream>>>(state_, field_, fieldW_, fieldH_, marginX_, marginY_,
                                             width_, height_, stride_, iterView_.scale,
                                             iterView_.refOffX, iterView_.refOffY,
                                             iterView_.maxIter, sliceIters_, sliceStart_ns_,
                                             50ull * 1000000ull, ref_, bla_, activeCount_);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord((cudaEvent_t)evStop_, stream);
    CUDA_CHECK(cudaMemcpyAsync(activeCountHost_, activeCount_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    cudaEventRecord((cudaEvent_t)evSlice_, stream);
    sliceStart_ += sliceIters_;
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
                                                lastImage_, devPtr, width_, height_, params,
                                                timeSec, map);
    err = cudaGetLastError();
    if (map.snapshot) {
        cudaMemcpyAsync(lastImage_, devPtr, sizeof(uint32_t) * (size_t)width_ * height_,
                        cudaMemcpyDeviceToDevice, stream);
    }
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
