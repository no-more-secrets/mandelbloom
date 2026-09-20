#pragma once
#include <cstdint>
#include "field.h"

// The pixel grid of the render pass and how it maps onto the complex
// plane. The centre itself lives on the host (high precision); the kernel
// only sees pixel offsets from the reference orbit. scale is complex units
// per pixel.
struct ViewParams {
    double scale = 3.0 / 800.0;
    // View centre minus reference centre, complex units. Non-zero after pans
    // that kept the old reference orbit.
    double refOffX = 0.0, refOffY = 0.0;
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

// How the display view maps onto the two image sources. A display pixel p
// lands on source pixel (ox, oy) + (p - centre) * ratio.
struct CompositeMap {
    double nox = 0, noy = 0, ratioN = 1;  // running pass's field
    double oox = 0, ooy = 0, ratioO = 1;  // last finished frame
    float newDetail = 0.f;  // source pixels per display pixel, 0 = unusable
    float oldDetail = 0.f;
    float pixelScaleN = 1.f;  // field's complex units per field pixel
    int snapshot = 0;         // keep the result as the new "last frame"
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
    // (Re)register the GL pixel unpack buffer and size the field to match,
    // plus a margin around the view so pans land on existing coarse data.
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
    // Pan: shift field and state by (dx, dy) pixels (content moves by
    // -dx, -dy), keep finished pixels, mark exposed strips pending, and
    // continue the pass coarse-to-fine on what is pending. Same reference.
    bool shiftAndResume(const ViewParams& view, int dx, int dy);
    // Reference changed: restart only the pixels that are not finished.
    bool restartPending(const ViewParams& view);
    bool stepIterate();
    bool iterateBusy();
    bool iterateDone() const { return iterDone_; }
    int iterateProgress() const { return sliceStart_; }  // iterations issued so far
    // Coarse-to-fine: the pass runs at stride 8, 4, 2, 1. completedStride is
    // the finest stride whose pixels are all finished (0 = none yet).
    int currentStride() const { return stride_; }
    int completedStride() const { return completedStride_; }
    // True once after each slice completes (cleared by the call).
    bool takeSliceFinished() { bool f = sliceFinished_; sliceFinished_ = false; return f; }

    // Display pass: colour the running field and the last finished frame
    // into the bound pixel buffer through the given mappings. Runs on its
    // own stream so it never waits for a slice.
    bool composite(const ShadeParams& params, float timeSec, const CompositeMap& map);

    float lastIterateMs() const { return iterateMs_; }  // total for the last full pass
    float lastSliceMs() const { return sliceMs_; }
    float lastShadeMs() const { return shadeMs_; }
    const char* deviceName() const { return deviceName_; }

private:
    void unregisterPbo();
    void freeField();
    void freeReference();
    void freeBla();
    void syncAll();
    void resetPass(const ViewParams& view);

    struct cudaGraphicsResource* pboResource_ = nullptr;
    FieldSample* field_ = nullptr;
    struct PixelState* state_ = nullptr;
    FieldSample* fieldAlt_ = nullptr;
    struct PixelState* stateAlt_ = nullptr;
    uint32_t* lastImage_ = nullptr;  // copy of the last finished frame
    double* refZr_ = nullptr;
    double* refZi_ = nullptr;
    int refCapacity_ = 0;
    DeviceReference ref_;
    struct BlaNode* blaNodes_ = nullptr;
    int* blaOffsets_ = nullptr;
    int blaNodeCapacity_ = 0;
    int blaLevelCapacity_ = 0;
    DeviceBla bla_;
    int* activeCount_ = nullptr;      // device
    unsigned long long* sliceStart_ns_ = nullptr;  // device, GPU clock at slice start
    int* activeCountHost_ = nullptr;  // pinned
    void* stream_ = nullptr;      // iteration
    void* dispStream_ = nullptr;  // composite
    void* evSlice_ = nullptr;
    ViewParams iterView_;
    int sliceStart_ = 0;
    int sliceIters_ = 512;
    int stride_ = 1;
    int completedStride_ = 0;
    int firstStride_ = 8;
    bool iterDone_ = true;
    bool sliceInFlight_ = false;
    bool sliceFinished_ = false;
    void* evShadeA_ = nullptr;
    void* evShadeB_ = nullptr;
    bool shadePending_ = false;
    int width_ = 0, height_ = 0;    // view (pixel buffer) size
    int fieldW_ = 0, fieldH_ = 0;   // field size = view + margins
    int marginX_ = 0, marginY_ = 0; // margin each side, computed at stride >= 2 only
    float iterateMs_ = 0.f, shadeMs_ = 0.f;
    float sliceMs_ = 0.f;
    float passMs_ = 0.f;
    char deviceName_[256] = "none";
    void* evStart_ = nullptr;
    void* evStop_ = nullptr;
};
