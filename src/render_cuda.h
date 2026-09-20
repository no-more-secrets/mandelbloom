#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
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
    bool useFloat = true;  // float kernel with per-value exponents (default); else double
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
    int gen = 0;             // generation id as stored in FieldSample::gen
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
    // Import the D3D12 shared buffer that receives RGBA16F output (rows of
    // rowPitchBytes) and size the field to match, plus a margin around the
    // view so pans land on existing coarse data. ss: supersampling factor;
    // the field's view region is (width*ss) x (height*ss) and the composite
    // averages ss*ss samples per pixel.
    bool bindOutput(void* sharedHandle, size_t sharedSize, int rowPitchBytes, int width,
                    int height, int ss);
    // Linear multiplier applied when writing output (SDR white level in
    // scRGB: 1.0 on SDR displays, e.g. 2.5 for a 200-nit SDR white in HDR),
    // and the display's headroom above it (peak / SDR white).
    void setOutputScale(float s, float headroom) { outScale_ = s; headroom_ = headroom; }
    // Post-process the linear image written by composite()/shadeCached()
    // into the output buffer. Must run after either before presenting.
    bool postProcess(const PostParams& params, uint32_t frame);
    // Wait for the display stream, so D3D12 may copy the output buffer.
    bool syncDisplay();
    // Read the output back as 8-bit sRGB (divided by the output scale).
    bool readOutput(std::vector<uint32_t>& rgba8);
    // Upload a new reference orbit (host arrays of `length` doubles).
    bool uploadReference(const double* zr, const double* zi, int length, bool escaped);
    // Upload a BLA table built for the current reference and view.
    bool uploadBla(const struct BlaNode* nodes, const struct BlaNodeF* nodesF, int count,
                   const int* levelOffset, int levels, int steps);

    // Heavy pass, run in slices so the UI stays live and no kernel runs long
    // enough to trip the Windows GPU watchdog. beginIterate resets state;
    // call stepIterate whenever !iterateBusy() until iterateDone().
    // gen: the id written into samples this pass produces (> 0, increasing).
    bool beginIterate(const ViewParams& view, int gen);
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

    // Settled path. buildPaletteLut tabulates one palette cycle (call when
    // palette parameters change). buildShadeCache resolves every subsample
    // once through the generation search; shadeCached then colours the
    // cache for a given time: a streaming pass with no field access.
    bool buildPaletteLut(const ShadeParams& params);
    bool buildShadeCache(const ShadeParams& params, const CompositeMap& map);
    bool shadeCached(const ShadeParams& params, float timeSec);

    // Debug: read the view region back and count pixels by state. Slow.
    struct DebugStats {
        int total = 0, genMatch = 0, genZero = 0, genOther = 0;
        int stActive = 0, stEscaped = 0, stInside = 0;
        int genMatchInside = 0;  // gen matches and iter < 0
    };
    bool debugStats(int gen, DebugStats& out);

    float lastIterateMs() const { return iterateMs_; }  // total for the last full pass
    float lastSliceMs() const { return sliceMs_; }
    float lastShadeMs() const { return shadeMs_; }
    float lastPostMs() const { return postMs_; }
    const char* deviceName() const { return deviceName_; }

private:
    void freeOutput();
    void freeField();
    void freeReference();
    void freeBla();
    void syncAll();
    void resetPass(const ViewParams& view);

    cudaExternalMemory_t extMem_ = nullptr;
    uint16_t* out_ = nullptr;   // mapped shared buffer, RGBA16F
    int outPitchPx_ = 0;        // pixels per row in out_
    float outScale_ = 1.f;
    float headroom_ = 1.f;
    uint16_t* hdr_ = nullptr;      // linear RGBA16F, display size, written by shading
    float4* bloomA_ = nullptr;     // 1/4 res
    float4* bloomT_ = nullptr;     // 1/4 res temp
    float4* bloomB_ = nullptr;     // 1/8 res
    float4* bloomT2_ = nullptr;    // 1/8 res temp
    float postMs_ = 0.f;
    FieldSample* field_ = nullptr;
    struct PixelState* state_ = nullptr;
    FieldSample* fieldAlt_ = nullptr;
    struct ShadeInput* shadeCache_ = nullptr;  // w*h*ss*ss entries
    float4* paletteLut_ = nullptr;
    struct PixelState* stateAlt_ = nullptr;
    double* refZr_ = nullptr;
    double* refZi_ = nullptr;
    int refCapacity_ = 0;
    DeviceReference ref_;
    struct BlaNode* blaNodes_ = nullptr;
    struct BlaNodeF* blaNodesF_ = nullptr;
    float2* refF_ = nullptr;  // float2 copy of the reference orbit
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
    int gen_ = 0;
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
