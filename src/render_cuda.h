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
    bool useBla = true;
};

// BLA table as uploaded to the device (see bla.h for the layout).
struct DeviceBla {
    const struct BlaNode* nodes = nullptr;
    const int* levelOffset = nullptr;  // levels entries
    int levels = 0;
    int steps = 0;
    int enabled = 0;
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
    // Upload a BLA table built for the current reference and view.
    bool uploadBla(const struct BlaNode* nodes, int count, const int* levelOffset, int levels,
                   int steps);
    // Heavy pass, run in slices so the UI stays live and no kernel runs long
    // enough to trip the Windows GPU watchdog. beginIterate resets state;
    // call stepIterate whenever !iterateBusy() until iterateDone().
    bool beginIterate(const ViewParams& view);
    bool stepIterate();
    bool iterateBusy();
    bool iterateDone() const { return iterDone_; }
    // True once after each slice completes (cleared by the call).
    bool takeSliceFinished() { bool f = sliceFinished_; sliceFinished_ = false; return f; }
    int iterateProgress() const { return sliceStart_; }  // iterations issued so far
    // Light pass: color the field into the bound pixel buffer.
    bool shade(const ShadeParams& params, float timeSec, double pixelScale);

    float lastIterateMs() const { return iterateMs_; }  // total for the last full pass
    float lastSliceMs() const { return sliceMs_; }
    float lastShadeMs() const { return shadeMs_; }
    const char* deviceName() const { return deviceName_; }

private:
    void unregisterPbo();
    void freeField();
    void freeReference();
    void freeBla();
    struct cudaGraphicsResource* pboResource_ = nullptr;
    FieldSample* field_ = nullptr;
    struct PixelState* state_ = nullptr;
    int* activeCount_ = nullptr;      // device
    int* activeCountHost_ = nullptr;  // pinned
    void* stream_ = nullptr;
    void* evSlice_ = nullptr;
    ViewParams iterView_;
    int sliceStart_ = 0;
    int sliceIters_ = 256;
    bool iterDone_ = true;
    bool sliceInFlight_ = false;
    bool sliceFinished_ = false;
    void* evShadeA_ = nullptr;
    void* evShadeB_ = nullptr;
    bool shadePending_ = false;
    float sliceMs_ = 0.f;
    float passMs_ = 0.f;
    double* refZr_ = nullptr;
    double* refZi_ = nullptr;
    int refCapacity_ = 0;
    DeviceReference ref_;
    struct BlaNode* blaNodes_ = nullptr;
    int* blaOffsets_ = nullptr;
    int blaNodeCapacity_ = 0;
    int blaLevelCapacity_ = 0;
    DeviceBla bla_;
    int width_ = 0, height_ = 0;
    float iterateMs_ = 0.f, shadeMs_ = 0.f;
    char deviceName_[256] = "none";
    void* evStart_ = nullptr;
    void* evStop_ = nullptr;
};
