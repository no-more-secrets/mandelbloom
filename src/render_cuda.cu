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

// Milestone 1 iteration kernel: plain double precision, with the
// derivative dz/dc carried along for the distance estimate. Placeholder
// until the perturbation path lands.
__global__ void iterateDouble(FieldSample* __restrict__ field, int w, int h, double cx,
                              double cy, double scale, int maxIter) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const double cr = cx + ((double)x - 0.5 * w) * scale;
    const double ci = cy - ((double)y - 0.5 * h) * scale;

    double zr = 0.0, zi = 0.0;    // z
    double dr = 0.0, di = 0.0;    // dz/dc
    double zr2 = 0.0, zi2 = 0.0;
    int i = 0;
    const double bailout = 65536.0;
    while (i < maxIter && zr2 + zi2 < bailout) {
        // d' = 2 z d + 1
        const double ndr = 2.0 * (zr * dr - zi * di) + 1.0;
        const double ndi = 2.0 * (zr * di + zi * dr);
        dr = ndr;
        di = ndi;
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        zr2 = zr * zr;
        zi2 = zi * zi;
        ++i;
    }

    FieldSample s;
    if (i >= maxIter) {
        s.iter = -1.f;
        s.de = 0.f;
        s.angle = 0.f;
    } else {
        const double mag2 = zr2 + zi2;
        const double logMag = 0.5 * log(mag2);
        // Continuous escape time: mu = i + 1 - log2(log|z|)
        s.iter = (float)(i + 1.0 - log2(logMag / 0.6931471805599453));
        // Distance estimate: |z| log|z| / |dz/dc|
        const double dmag = sqrt(dr * dr + di * di);
        s.de = dmag > 0.0 ? (float)(sqrt(mag2) * logMag / dmag) : 0.f;
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
    const dim3 block(16, 16);
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    cudaEventRecord((cudaEvent_t)evStart_, 0);
    iterateDouble<<<grid, block>>>(field_, width_, height_, view.cx, view.cy, view.scale,
                                   view.maxIter);
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
