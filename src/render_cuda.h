#pragma once
#include <cstdint>
#include "field.h"

// The pixel grid and how it maps onto the complex plane. The centre
// itself lives in the reference orbit (high precision); the kernel only
// sees pixel offsets from it. scale is complex units per pixel.
struct ViewParams {
    double scale = 3.0 / 800.0;
    int width = 0;
    int height = 0;
    int maxIter = 512;
};

// Reference orbit as uploaded to the device.
struct DeviceReference {
    const double* zr = nullptr;  // device pointers, length entries each
    const double* zi = nullptr;
    int length = 0;
    bool escaped = false;
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
    // Upload a new reference orbit (host arrays of `length` doubles).
    bool uploadReference(const double* zr, const double* zi, int length, bool escaped);
    // Heavy pass: fill the field for this view around the reference.
    bool iterate(const ViewParams& view);
    // Light pass: color the field into the bound pixel buffer.
    bool shade(const ShadeParams& params, float timeSec, double pixelScale);

    float lastIterateMs() const { return iterateMs_; }
    float lastShadeMs() const { return shadeMs_; }
    const char* deviceName() const { return deviceName_; }

private:
    void unregisterPbo();
    void freeField();
    void freeReference();
    struct cudaGraphicsResource* pboResource_ = nullptr;
    FieldSample* field_ = nullptr;
    double* refZr_ = nullptr;
    double* refZi_ = nullptr;
    int refCapacity_ = 0;
    DeviceReference ref_;
    int width_ = 0, height_ = 0;
    float iterateMs_ = 0.f, shadeMs_ = 0.f;
    char deviceName_[256] = "none";
    void* evStart_ = nullptr;
    void* evStop_ = nullptr;
};
