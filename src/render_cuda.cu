#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cmath>
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

__global__ void resetState(PixelState* __restrict__ st, FieldSample* __restrict__ field,
                           int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    st[i] = PixelState{};
    FieldSample s{};
    s.iter = 0.f;
    s.pad = 1.f;  // pending
    field[i] = s;
}

__device__ __forceinline__ void writeSample(FieldSample* __restrict__ field, size_t idx,
                                            const PixelState& st, double zr, double zi,
                                            double scale, bool escaped, bool pending) {
    FieldSample s;
    if (!escaped) {
        s.iter = pending ? (float)st.n : -1.f;
        s.de = 0.f;
        s.angle = 0.f;
    } else {
        const double mag2 = zr * zr + zi * zi;
        const double logMag = 0.5 * log(mag2);
        s.iter = (float)(st.n + 1.0 - log2(logMag / 0.6931471805599453));
        const double dmag = sqrt(st.dr * st.dr + st.di * st.di);
        s.de = dmag > 0.0 ? (float)(sqrt(mag2) * logMag / dmag * scale) : 0.f;
        s.angle = (float)atan2(zi, zr);
    }
    s.pad = pending ? 1.f : 0.f;
    field[idx] = s;
}

// Perturbation iteration in double precision with Zhuoran's rebasing,
// advanced by at most sliceIters iterations per launch.
//   dz_{n+1} = 2 Z_m dz_n + dz_n^2 + dc
//   rebase when |Z_m + dz| < |dz|: dz = Z_m + dz, m = 0
// Find the longest BLA node starting at reference index m that is valid
// for |dz|^2 = dzmag2. Returns nullptr if none.
__device__ __forceinline__ const BlaNode* findBla(const DeviceBla& bla, int m, double dzmag2,
                                                  int refLast) {
    const int j0 = m - 1;
    if (j0 < 0 || j0 >= bla.steps) return nullptr;
    // Highest level whose node is aligned at j0.
    int k = j0 == 0 ? bla.levels - 1 : min(bla.levels - 1, __ffs(j0) - 1);
    for (; k >= 0; --k) {
        const BlaNode* n = bla.nodes + bla.levelOffset[k] + (j0 >> k);
        if (dzmag2 < n->r2 && m + n->l <= refLast) return n;
    }
    return nullptr;
}

__device__ __forceinline__ unsigned long long globalTimerNs() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Stamps the GPU clock right before the slice so the deadline is relative
// to when the GPU actually started it, not when the host queued it.
__global__ void stampTimer(unsigned long long* __restrict__ out) { *out = globalTimerNs(); }

__global__ void iterateSlice(PixelState* __restrict__ state, FieldSample* __restrict__ field,
                             int w, int h, int stride, double scale, int maxIter, int sliceIters,
                             const unsigned long long* __restrict__ startNs,
                             unsigned long long budgetNs, DeviceReference ref, DeviceBla bla,
                             int* __restrict__ activeCount) {
    const unsigned long long deadlineNs = *startNs + budgetNs;
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    const int y = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
    const bool inBounds = x < w && y < h;
    const size_t idx = inBounds ? (size_t)y * w + x : 0;
    PixelState st{};
    if (inBounds) st = state[idx];
    bool active = inBounds && st.status == 0;

    if (active) {
        const double dcr = ((double)x - 0.5 * w) * scale;
        const double dci = -((double)y - 0.5 * h) * scale;
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
            // Hard wall-clock budget so a slice can never freeze the display.
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

__global__ void shadeKernel(const FieldSample* __restrict__ field, uint32_t* __restrict__ out,
                            int w, int h, ShadeParams p, float timeSec, float pixelScale,
                            int fillStride) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t idx = (size_t)y * w + x;
    FieldSample s = field[idx];
    if (s.pad > 0.5f && fillStride > 1) {
        // Not computed yet: borrow the finished coarse sample for this block.
        const int bx = x - (x % fillStride), by = y - (y % fillStride);
        s = field[(size_t)by * w + bx];
    }
    out[idx] = packSRGB8(shadeSample(s, p, timeSec, pixelScale));
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
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    stream_ = st;
    CUDA_CHECK(cudaMalloc(&activeCount_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sliceStart_ns_, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocHost(&activeCountHost_, sizeof(int)));
    *activeCountHost_ = 0;
    return true;
}

void CudaRenderer::unregisterPbo() {
    if (pboResource_) {
        cudaGraphicsUnregisterResource(pboResource_);
        pboResource_ = nullptr;
    }
}

void CudaRenderer::freeField() {
    if (stream_) cudaStreamSynchronize((cudaStream_t)stream_);
    if (field_) cudaFree(field_);
    if (state_) cudaFree(state_);
    field_ = nullptr;
    state_ = nullptr;
    iterDone_ = true;
    sliceInFlight_ = false;
}

void CudaRenderer::freeReference() {
    if (stream_) cudaStreamSynchronize((cudaStream_t)stream_);
    if (refZr_) cudaFree(refZr_);
    if (refZi_) cudaFree(refZi_);
    refZr_ = refZi_ = nullptr;
    refCapacity_ = 0;
    ref_ = DeviceReference{};
}

void CudaRenderer::freeBla() {
    if (stream_) cudaStreamSynchronize((cudaStream_t)stream_);
    if (blaNodes_) cudaFree(blaNodes_);
    if (blaOffsets_) cudaFree(blaOffsets_);
    blaNodes_ = nullptr;
    blaOffsets_ = nullptr;
    blaNodeCapacity_ = blaLevelCapacity_ = 0;
    bla_ = DeviceBla{};
}

bool CudaRenderer::uploadBla(const BlaNode* nodes, int count, const int* levelOffset, int levels,
                             int steps) {
    cudaStream_t stream = (cudaStream_t)stream_;
    CUDA_CHECK(cudaStreamSynchronize(stream));
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
    cudaStream_t stream = (cudaStream_t)stream_;
    CUDA_CHECK(cudaStreamSynchronize(stream));  // no slice may still read the old orbit
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
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&pboResource_, glPbo,
                                            cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK(cudaMalloc(&field_, sizeof(FieldSample) * (size_t)width * height));
    CUDA_CHECK(cudaMalloc(&state_, sizeof(PixelState) * (size_t)width * height));
    return true;
}

bool CudaRenderer::beginIterate(const ViewParams& view) {
    if (!field_ || !state_ || view.width != width_ || view.height != height_) return false;
    if (ref_.length < 2) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    // Cancel whatever slice is in flight (bounded by one slice length).
    CUDA_CHECK(cudaStreamSynchronize(stream));
    iterView_ = view;
    bla_.enabled = view.useBla ? 1 : 0;
    sliceStart_ = 0;
    sliceIters_ = 512;  // modest start each pass; the GPU deadline bounds it anyway
    stride_ = firstStride_;
    completedStride_ = 0;
    iterDone_ = false;
    sliceInFlight_ = false;
    passMs_ = 0.f;
    const int count = width_ * height_;
    resetState<<<(count + 255) / 256, 256, 0, stream>>>(state_, field_, count);
    CUDA_CHECK(cudaGetLastError());
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
    const int sw = (width_ + stride_ - 1) / stride_, sh = (height_ + stride_ - 1) / stride_;
    const dim3 grid((sw + block.x - 1) / block.x, (sh + block.y - 1) / block.y);
    CUDA_CHECK(cudaMemsetAsync(activeCount_, 0, sizeof(int), stream));
    cudaEventRecord((cudaEvent_t)evStart_, stream);
    stampTimer<<<1, 1, 0, stream>>>(sliceStart_ns_);
    iterateSlice<<<grid, block, 0, stream>>>(state_, field_, width_, height_, stride_,
                                             iterView_.scale, iterView_.maxIter, sliceIters_,
                                             sliceStart_ns_,
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

bool CudaRenderer::shade(const ShadeParams& params, float timeSec, double pixelScale,
                         int fillStride) {
    if (!pboResource_ || !field_) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
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
    // Timing of the previous shade, harvested without blocking.
    if (shadePending_ && cudaEventQuery((cudaEvent_t)evShadeB_) == cudaSuccess) {
        cudaEventElapsedTime(&shadeMs_, (cudaEvent_t)evShadeA_, (cudaEvent_t)evShadeB_);
        shadePending_ = false;
    }
    if (!shadePending_) cudaEventRecord((cudaEvent_t)evShadeA_, stream);
    shadeKernel<<<grid, block, 0, stream>>>(field_, devPtr, width_, height_, params, timeSec,
                                            (float)pixelScale, fillStride);
    err = cudaGetLastError();
    if (!shadePending_) {
        cudaEventRecord((cudaEvent_t)evShadeB_, stream);
        shadePending_ = true;
    }
    // Unmap is stream-ordered; later GL calls wait for it without a host sync.
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &pboResource_, stream));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "shade kernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}
