#pragma once
#include <cstdint>

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

// Owns the CUDA side of the pipeline: interop registration of the GL
// pixel buffer, kernel launch, and timing.
class CudaRenderer {
public:
    CudaRenderer() = default;
    ~CudaRenderer();
    CudaRenderer(const CudaRenderer&) = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;

    // Call after the GL context is current. Returns false on failure.
    bool init();
    // (Re)register the GL pixel unpack buffer that receives the image.
    bool bindPixelBuffer(unsigned glPbo, int width, int height);
    // Render the view into the bound buffer. Blocks until done.
    bool render(const ViewParams& view);

    float lastRenderMs() const { return lastMs_; }
    const char* deviceName() const { return deviceName_; }

private:
    void unregisterPbo();
    struct cudaGraphicsResource* pboResource_ = nullptr;
    int width_ = 0, height_ = 0;
    float lastMs_ = 0.f;
    char deviceName_[256] = "none";
    void* evStart_ = nullptr;
    void* evStop_ = nullptr;
};
