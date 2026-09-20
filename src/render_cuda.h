#pragma once
#include <cstdint>
#include "field.h"

// View of the complex plane mapped onto a pixel grid.
// scale is complex units per pixel. Pixel (0,0) is top-left.
struct ViewParams {
    double cx = -0.5;
    double cy = 0.0;
    double scale = 3.0 / 800.0;
    int width = 0;
    int height = 0;
    int maxIter = 512;
};

// Owns the CUDA side: the iteration field, the interop registration of
// the GL pixel buffer, kernel launches, and timing.
class CudaRenderer {
public:
    CudaRenderer() = default;
    ~CudaRenderer();
    CudaRenderer(const CudaRenderer&) = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;

    // Call after the GL context is current. Returns false on failure.
    bool init();
    // (Re)register the GL pixel unpack buffer and size the field to match.
    bool bindPixelBuffer(unsigned glPbo, int width, int height);
    // Heavy pass: fill the field for this view. Blocks until done.
    bool iterate(const ViewParams& view);
    // Light pass: color the field into the bound pixel buffer.
    bool shade(const ShadeParams& params, float timeSec, double pixelScale);

    float lastIterateMs() const { return iterateMs_; }
    float lastShadeMs() const { return shadeMs_; }
    const char* deviceName() const { return deviceName_; }

private:
    void unregisterPbo();
    void freeField();
    struct cudaGraphicsResource* pboResource_ = nullptr;
    FieldSample* field_ = nullptr;
    int width_ = 0, height_ = 0;
    float iterateMs_ = 0.f, shadeMs_ = 0.f;
    char deviceName_[256] = "none";
    void* evStart_ = nullptr;
    void* evStop_ = nullptr;
};
