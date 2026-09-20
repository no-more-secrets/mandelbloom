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

// One render generation still present in the field: how the display view
// maps onto the field through the view that generation was rendered with.
// A display pixel p lands on field pixel (ox, oy) + (p - centre) * ratio,
// in view-region coordinates (the kernel adds the margin).
struct GenMap {
    double ox = 0, oy = 0, ratio = 1;
    float pixelScale = 1.f;  // complex units per field pixel for this generation
    float gen = 0.f;         // generation id as stored in FieldSample::gen
};

#define MAX_GENS 8

struct CompositeMap {
    GenMap gens[MAX_GENS];  // newest first
    int genCount = 0;
    int ss = 1;             // subsamples per axis averaged per display pixel
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
    // ss: supersampling factor; the field's view region is (width*ss) x
    // (height*ss) and the composite averages ss*ss samples per pixel.
    bool bindPixelBuffer(unsigned glPbo, int width, int height, int ss);
    // Upload a new reference orbit (host arrays of `length` doubles).
    bool uploadReference(const double* zr, const double* zi, int length, bool escaped);
    // Upload a BLA table built for the current reference and view.
    bool uploadBla(const struct BlaNode* nodes, int count, const int* levelOffset, int levels,
                   int steps);

    // Heavy pass, run in slices so the UI stays live and no kernel runs long
    // enough to trip the Windows GPU watchdog. beginIterate resets state;
    // call stepIterate whenever !iterateBusy() until iterateDone().
    // gen: the id written into samples this pass produces (> 0, increasing).
    bool beginIterate(const ViewParams& view, float gen);
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

    // Display pass: colour the field into the bound pixel buffer, taking
    // each pixel from the best generation that has data for it. Runs on its
    // own stream so it never waits for a slice.
    bool composite(const ShadeParams& params, float timeSec, const CompositeMap& map);

    // Debug: read the view region back and count pixels by state. Slow.
    struct DebugStats {
        int total = 0, genMatch = 0, genZero = 0, genOther = 0;
        int stActive = 0, stEscaped = 0, stInside = 0;
        int genMatchInside = 0;  // gen matches and iter < 0
    };
    bool debugStats(float gen, DebugStats& out);

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
    float gen_ = 0.f;
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
    int width_ = 0, height_ = 0;    // display (pixel buffer) size
    int viewW_ = 0, viewH_ = 0;     // view region in field pixels (display * ss)
    int ss_ = 1;
    int fieldW_ = 0, fieldH_ = 0;   // field size = view + margins
    int marginX_ = 0, marginY_ = 0; // margin each side, computed at stride >= 2 only
    float iterateMs_ = 0.f, shadeMs_ = 0.f;
    float sliceMs_ = 0.f;
    float passMs_ = 0.f;
    char deviceName_[256] = "none";
    void* evStart_ = nullptr;
    void* evStop_ = nullptr;
};
