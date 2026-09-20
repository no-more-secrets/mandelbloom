#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cmath>
#include "render_cuda.h"

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

__device__ __forceinline__ uint32_t packRGBA(float r, float g, float b) {
    r = fminf(fmaxf(r, 0.f), 1.f);
    g = fminf(fmaxf(g, 0.f), 1.f);
    b = fminf(fmaxf(b, 0.f), 1.f);
    uint32_t ir = (uint32_t)(r * 255.f + 0.5f);
    uint32_t ig = (uint32_t)(g * 255.f + 0.5f);
    uint32_t ib = (uint32_t)(b * 255.f + 0.5f);
    return ir | (ig << 8) | (ib << 16) | 0xFF000000u;
}

// Cosine palette. t in iterations, period 64.
__device__ __forceinline__ uint32_t palette(float t) {
    const float k = 6.2831853f / 64.f;
    float r = 0.5f + 0.5f * cosf(k * t + 0.0f);
    float g = 0.5f + 0.5f * cosf(k * t + 2.1f);
    float b = 0.5f + 0.5f * cosf(k * t + 4.2f);
    return packRGBA(r, g, b);
}

// Milestone 1 kernel: plain double-precision escape time with smooth
// coloring. Placeholder until the perturbation path lands.
__global__ void mandelDouble(uint32_t* __restrict__ out, int w, int h,
                             double cx, double cy, double scale, int maxIter) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const double cr = cx + ((double)x - 0.5 * w) * scale;
    const double ci = cy - ((double)y - 0.5 * h) * scale;

    double zr = 0.0, zi = 0.0, zr2 = 0.0, zi2 = 0.0;
    int i = 0;
    const double bailout = 65536.0;
    while (i < maxIter && zr2 + zi2 < bailout) {
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        zr2 = zr * zr;
        zi2 = zi * zi;
        ++i;
    }

    uint32_t px;
    if (i >= maxIter) {
        px = 0xFF000000u;
    } else {
        // Continuous escape time: mu = i + 1 - log2(log2|z|)
        float logzn = 0.5f * logf((float)(zr2 + zi2));
        float nu = log2f(logzn / 0.6931472f);
        float mu = (float)i + 1.f - nu;
        px = palette(mu);
    }
    out[(size_t)y * w + x] = px;
}

}  // namespace

CudaRenderer::~CudaRenderer() {
    unregisterPbo();
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

bool CudaRenderer::bindPixelBuffer(unsigned glPbo, int width, int height) {
    unregisterPbo();
    width_ = width;
    height_ = height;
    if (!glPbo || width <= 0 || height <= 0) return true;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&pboResource_, glPbo,
                                            cudaGraphicsRegisterFlagsWriteDiscard));
    return true;
}

bool CudaRenderer::render(const ViewParams& view) {
    if (!pboResource_ || view.width != width_ || view.height != height_) return false;

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
    mandelDouble<<<grid, block>>>(devPtr, width_, height_, view.cx, view.cy, view.scale,
                                  view.maxIter);
    cudaEventRecord((cudaEvent_t)evStop_, 0);
    err = cudaGetLastError();
    cudaEventSynchronize((cudaEvent_t)evStop_);
    cudaEventElapsedTime(&lastMs_, (cudaEvent_t)evStart_, (cudaEvent_t)evStop_);
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &pboResource_, 0));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}
