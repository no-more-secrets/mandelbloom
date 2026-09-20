#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cmath>
#include "render_cuda.h"
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

namespace {

// Perturbation iteration in double precision with Zhuoran's rebasing.
// Pixel parameter c = C + dc where C is the reference centre. We track
// dz = z - Z_m and the derivative dz/dc (scaled by pixel size) for the
// distance estimate.
//   dz_{n+1} = 2 Z_m dz_n + dz_n^2 + dc
//   rebase when |Z_m + dz| < |dz|: dz = Z_m + dz, m = 0
__global__ void iteratePerturb(FieldSample* __restrict__ field, int w, int h, double scale,
                               int maxIter, DeviceReference ref) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const double dcr = ((double)x - 0.5 * w) * scale;
    const double dci = -((double)y - 0.5 * h) * scale;

    double dzr = 0.0, dzi = 0.0;  // dz
    double zr = 0.0, zi = 0.0;    // full z = Z_m + dz
    double dr = 0.0, di = 0.0;    // d(z)/d(c) * scale
    int m = 0;                    // reference index
    int n = 0;                    // pixel iteration
    const double bailout = 65536.0;
    const int refLast = ref.length - 1;
    bool escaped = false;

    while (n < maxIter) {
        const double Zr = ref.zr[m];
        const double Zi = ref.zi[m];
        // derivative uses full z_n = Z_m + dz_n (before the step)
        const double fzr = Zr + dzr, fzi = Zi + dzi;
        const double ndr = 2.0 * (fzr * dr - fzi * di) + scale;
        const double ndi = 2.0 * (fzr * di + fzi * dr);
        dr = ndr;
        di = ndi;
        // dz' = (2 Z + dz) dz + dc
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
        if (zmag2 > bailout) { escaped = true; break; }
        const double dzmag2 = dzr * dzr + dzi * dzi;
        if (zmag2 < dzmag2 || m >= refLast) {
            dzr = zr;
            dzi = zi;
            m = 0;
        }
    }

    FieldSample s;
    if (!escaped) {
        s.iter = -1.f;
        s.de = 0.f;
        s.angle = 0.f;
    } else {
        const double mag2 = zr * zr + zi * zi;
        const double logMag = 0.5 * log(mag2);
        s.iter = (float)(n + 1.0 - log2(logMag / 0.6931471805599453));
        const double dmag = sqrt(dr * dr + di * di);
        // de here is already in pixels (derivative was scaled), convert to
        // complex units so the shader's pixelScale division still holds.
        s.de = dmag > 0.0 ? (float)(sqrt(mag2) * logMag / dmag * scale) : 0.f;
        s.angle = (float)atan2(zi, zr);
    }
    s.pad = 0.f;
    field[(size_t)y * w + x] = s;
}

__global__ void shadeKernel(const FieldSample* __restrict__ field, uint32_t* __restrict__ out,
                            int w, int h, ShadeParams p, float timeSec, float pixelScale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t idx = (size_t)y * w + x;
    out[idx] = packSRGB8(shadeSample(field[idx], p, timeSec, pixelScale));
}

}  // namespace

CudaRenderer::~CudaRenderer() {
    unregisterPbo();
    freeField();
    freeReference();
    if (evStart_) cudaEventDestroy((cudaEvent_t)evStart_);
    if (evStop_) cudaEventDestroy((cudaEvent_t)evStop_);
}

bool CudaRenderer::init() {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::snprintf(deviceName_, sizeof deviceName_, "%s (SM %d.%d, %d SMs)", prop.name,
                  prop.major, prop.minor, prop.multiProcessorCount);
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    evStart_ = a;
    evStop_ = b;
    return true;
}

void CudaRenderer::unregisterPbo() {
    if (pboResource_) {
        cudaGraphicsUnregisterResource(pboResource_);
        pboResource_ = nullptr;
    }
}

void CudaRenderer::freeField() {
    if (field_) {
        cudaFree(field_);
        field_ = nullptr;
    }
}

void CudaRenderer::freeReference() {
    if (refZr_) cudaFree(refZr_);
    if (refZi_) cudaFree(refZi_);
    refZr_ = refZi_ = nullptr;
    refCapacity_ = 0;
    ref_ = DeviceReference{};
}

bool CudaRenderer::uploadReference(const double* zr, const double* zi, int length, bool escaped) {
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
    return true;
}

bool CudaRenderer::iterate(const ViewParams& view) {
    if (!field_ || view.width != width_ || view.height != height_) return false;
    if (ref_.length < 2) return false;
    const dim3 block(16, 16);
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    cudaEventRecord((cudaEvent_t)evStart_, 0);
    iteratePerturb<<<grid, block>>>(field_, width_, height_, view.scale, view.maxIter, ref_);
    cudaEventRecord((cudaEvent_t)evStop_, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)evStop_));
    cudaEventElapsedTime(&iterateMs_, (cudaEvent_t)evStart_, (cudaEvent_t)evStop_);
    return true;
}

bool CudaRenderer::shade(const ShadeParams& params, float timeSec, double pixelScale) {
    if (!pboResource_ || !field_) return false;
    CUDA_CHECK(cudaGraphicsMapResources(1, &pboResource_, 0));
    uint32_t* devPtr = nullptr;
    size_t bytes = 0;
    cudaError_t err = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &bytes, pboResource_);
    if (err != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &pboResource_, 0);
        std::fprintf(stderr, "map failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    cudaEventRecord((cudaEvent_t)evStart_, 0);
    shadeKernel<<<grid, block>>>(field_, devPtr, width_, height_, params, timeSec,
                                 (float)pixelScale);
    cudaEventRecord((cudaEvent_t)evStop_, 0);
    err = cudaGetLastError();
    cudaEventSynchronize((cudaEvent_t)evStop_);
    cudaEventElapsedTime(&shadeMs_, (cudaEvent_t)evStart_, (cudaEvent_t)evStop_);
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &pboResource_, 0));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "shade kernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}
