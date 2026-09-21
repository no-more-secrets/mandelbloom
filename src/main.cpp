#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "bigfloat.h"
#include "bla.h"
#include "dx_display.h"
#include "image_io.h"
#include "script.h"
#include "presets.h"
#include "reference.h"
#include "render_cuda.h"
#include "video_encoder.h"

namespace {

// A view of the plane: centre at high precision, pixel scale.
struct View {
    BigFloat cx{64}, cy{64};
    double scale = 1.0;
};

// Zoom video export settings (UI and --video-* flags).
struct VideoParams {
    int width = 0, height = 0;   // 0: the window's pixel size
    double fps = 60.0;
    double seconds = 30.0;       // zooming time, between the holds
    double zoomFrom = 1.0;       // starting zoom (1 = the whole set)
    double holdStart = 1.0, holdEnd = 2.0;  // seconds held on the first / last frame
    float ease = 0.3f;           // 0: constant doublings per second, 1: smooth start and stop
    bool hdr = true;             // 10-bit PQ BT.2020 HEVC; else 8-bit BT.709 HEVC
    float nits = 0.f;            // SDR white in nits for HDR output; 0: the display's
    float headroom = 0.f;        // peak / SDR white for the tone map; 0: the display's
    int cq = 20;                 // NVENC constant quality
    int subsamples = 3;          // composite samples per axis per output pixel
    bool debug = false;          // print timing per second
    bool dumpKeys = false;       // write every keyframe as a PNG and print field statistics (slow)
    bool iterRamp = true;        // shallow keyframes get fewer iterations (2000 -> full by mid-depth)
    std::string codec;           // ffmpeg encoder; empty: hevc_nvenc
    std::string outPath;         // empty: Videos\Mandelbloom\mandel_<stamp>_<zoom>.mp4
};

// A running export. Keyframes are rendered 2x apart in zoom into two field
// slots; every output frame is composited from the finer keyframe in the
// centre and the coarser one around it.
struct VideoJob {
    bool active = false;
    bool cancel = false;
    bool quitAfter = false;
    bool previewOk = true;
    VideoParams p;
    VideoEncoder enc;
    std::string outPath, error;
    int w = 0, h = 0;
    float nits = 203.f, headroom = 1.f;
    View target;                        // centre and end scale (per output pixel)
    double startScale = 0, endScale = 0;
    int keyCount = 0, keysDone = 0;
    struct Key {
        int slot = 0, gen = 0;
        float minIter = -1.f;
        bool ready = false;
    };
    std::vector<Key> keys;
    int iterating = -1, nextKey = 0;    // keyframe in the live field, next to start
    int totalFrames = 0, frame = 0;
    double t0 = 0, keyMs = 0;
    double lastKeySec = 0, frameSec = 0;  // recent keyframe time, smoothed frame time
    double lastFrameStamp = 0;
    // Debug timing: wall seconds per phase since the last report, slices issued.
    double tPump = 0, tFrames = 0, tLive = 0, tPreview = 0, tReport = 0;
    int slices = 0, framesSince = 0;
    View savedRender, savedShown;
    ViewParams savedView;
};

struct App {
    SDL_Window* window = nullptr;
    DxDisplay display;
    // HDR state of the window's display, from SDL.
    bool hdrEnabled = false;
    float sdrWhite = 1.f;   // SDR white in linear scRGB units (1.0 = 80 nits)
    float hdrHeadroom = 1.f;
    bool vsync = true;
    CudaRenderer renderer;

    // render: where the pass computes (the target). shown: what the display
    // shows, tweened toward render. gens: the views of the generations whose
    // samples may still be in the field, newest first (gens[0] == render).
    ViewParams view;   // render pass grid + scale
    View render;
    View shown;
    struct Gen {
        View view;
        int id = 0;
        float minIter = -1.f;  // smallest escape iteration seen, -1 unknown
    };
    std::vector<Gen> gens;
    int nextGen = 1;
    int ss = 1;          // supersampling factor per axis (field = display * ss)
    AaParams aa;
    int ssApplied = 0;   // what the renderer is currently bound with
    bool tweening = false;
    float tweenTau = 0.08f;  // seconds to close ~63% of the gap
    bool tweenEnabled = true;
    // Zoom anchor: the plane point under the cursor stays put while the
    // scale glides. shown.centre = anchor - anchorPx * shown.scale.
    BigFloat anchorX{64}, anchorY{64};
    double anchorPxX = 0, anchorPxY = 0;

    BigFloat refCx{64}, refCy{64};  // reference orbit centre (kept across pans)
    int panDx = 0, panDy = 0;       // accumulated pan this frame, field pixels
    bool panDirty = false;
    ReferenceOrbit ref;
    int rerefRounds = 0;        // re-references done for the current generation
    int rerefCheckedStride = 0; // last completed stride checked for unreliable pixels
    BlaTable bla;
    std::vector<BlaNodeF> blaF;
    float blaEpsLog2 = -24.f;
    ShadeParams shade;
    PostParams post;
    PostParams lastPost;
    bool lastPostValid = false;
    uint32_t frameIndex = 0;
    bool imageDirty = false;  // shading wrote a new linear image this frame
    bool dirty = true;  // render view changed: restart the pass
    bool animate = false;
    float animTime = 0.f;  // seconds of animation elapsed (advances only while animating)
    // Settled display: cache of resolved subsamples + palette LUT, so an
    // animated but otherwise static view costs one streaming pass per frame.
    bool cacheValid = false;
    bool cachedShown = false;   // the pixel buffer currently holds a cached shade
    ShadeParams lastShade;      // what the cache / LUT / last shade were built with
    bool lastShadeValid = false;
    int cacheGen = -1;
    double lastShadeSec = 0.0;
    float bgFps = 0.f;          // cap for animated frames while settled; 0 = every frame
    bool dragging = false;
    // Pan inertia: velocity in pixels per second, carried after release.
    double velX = 0, velY = 0;
    double inertiaAccX = 0, inertiaAccY = 0;  // sub-pixel remainder
    double lastMotionSec = 0;
    bool inertia = false;
    bool inertiaEnabled = true;
    float inertiaTau = 0.25f;  // seconds for the velocity to fall to 37%
    bool showUi = true;
    bool fullscreen = false;
    int screenshotRequest = 0;   // 1 = PNG, 2 = PNG + EXR; handled after present
    std::string toast;
    double toastUntil = 0.0;
    std::string lastShotPath;
    char presetName[64] = "my preset";
    std::vector<std::string> savedPresets;
    bool savedPresetsValid = false;
    double fps = 0.0;
    float uiScale = 1.f;
    // Video export.
    VideoParams videoParams;
    int videoSizeChoice = 0;   // 0 window, 1 1080p, 2 1440p, 3 2160p
    bool autoVideo = false;    // --video: start once the window is bound
    VideoJob video;
    std::string lastVideoPath;
};

// Match ImGui's fonts and metrics to the monitor's content scale so the
// overlay obeys Windows display scaling.
void applyUiScale(App& app) {
    float s = SDL_GetWindowDisplayScale(app.window);
    if (s <= 0.f) s = 1.f;
    if (s == app.uiScale) return;
    app.uiScale = s;
    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle();
    ImGui::StyleColorsDark();
    style.ScaleAllSizes(s);
    style.FontScaleDpi = s;
    // Colours stay sRGB: the UI is drawn to an 8-bit texture and composited
    // in linear light at SDR white by the display.
}

void readHdrState(App& app) {
    const SDL_PropertiesID props = SDL_GetWindowProperties(app.window);
    app.hdrEnabled = SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false);
    app.sdrWhite = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT, 1.f);
    app.hdrHeadroom = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 1.f);
    if (app.sdrWhite <= 0.f) app.sdrWhite = 1.f;
    app.renderer.setOutputScale(app.sdrWhite, app.hdrHeadroom);
    app.uiScale = 0.f;  // re-apply style with the new white level
    applyUiScale(app);
}

// Bits needed so the centre is exact to well under a pixel.
mpfr_prec_t precisionFor(double scale) {
    const double bits = -std::log2(std::max(scale, 1e-300)) + 64.0;
    return (mpfr_prec_t)std::max(64.0, std::ceil(bits / 32.0) * 32.0);
}

void updatePrecision(App& app) {
    const mpfr_prec_t p = precisionFor(app.render.scale);
    if (p != app.render.cx.prec()) {
        app.render.cx.setPrec(p);
        app.render.cy.setPrec(p);
    }
    if (p > app.shown.cx.prec()) {
        app.shown.cx.setPrec(p);
        app.shown.cy.setPrec(p);
    }
}

void snapShown(App& app) {
    app.shown.cx = app.render.cx;
    app.shown.cy = app.render.cy;
    app.shown.scale = app.render.scale;
    app.tweening = false;
}

// Keep the render-pass parameters in step with the display-space view.
void syncViewScale(App& app) { app.view.scale = app.render.scale / app.ss; }

void zoomAt(App& app, float mx, float my, double factor) {
    // The point under the cursor in the view the user is looking at (the
    // shown view) is the anchor. The target scale accumulates; the target
    // centre is whatever keeps the anchor under the cursor at that scale.
    // Cursor offset from the centre in display pixels (the view is ss times larger).
    const double hx = 0.5 * app.view.width / app.ss, hy = 0.5 * app.view.height / app.ss;
    const double px = mx - hx, py = my - hy;
    app.render.scale *= factor;
    syncViewScale(app);
    updatePrecision(app);
    app.anchorX = app.shown.cx;
    app.anchorY = app.shown.cy;
    app.anchorX.setPrec(app.render.cx.prec());
    app.anchorY.setPrec(app.render.cy.prec());
    app.anchorX.addDouble(px * app.shown.scale);
    app.anchorY.subDouble(py * app.shown.scale);
    app.anchorPxX = px;
    app.anchorPxY = py;
    app.render.cx = app.anchorX;
    app.render.cy = app.anchorY;
    app.render.cx.subDouble(px * app.render.scale);
    app.render.cy.addDouble(py * app.render.scale);
    app.dirty = true;
    if (app.tweenEnabled) app.tweening = true;
    else snapShown(app);
}

void resetView(App& app) {
    app.render.scale = 3.2 / std::max(1, app.view.width / app.ss);
    syncViewScale(app);
    updatePrecision(app);
    app.render.cx.set(-0.5);
    app.render.cy.set(0.0);
    snapShown(app);
    app.dirty = true;
}

double nowSeconds() {
    return (double)SDL_GetPerformanceCounter() / (double)SDL_GetPerformanceFrequency();
}

// Pan the render (and shown) view by whole pixels; the field shifts to match.
void panPixels(App& app, int dx, int dy) {
    if (dx == 0 && dy == 0) return;
    if (app.tweening) snapShown(app);  // a pan is 1:1, cut any zoom tween short
    app.render.cx.subDouble(dx * app.render.scale);
    app.render.cy.addDouble(dy * app.render.scale);
    app.shown.cx = app.render.cx;
    app.shown.cy = app.render.cy;
    // Content moves with the mouse: new pixel x shows old pixel x - dx.
    app.panDx -= dx * app.ss;
    app.panDy -= dy * app.ss;
    app.panDirty = true;
}

// Coast after a drag, slowing with friction.
void applyInertia(App& app, double dt) {
    if (!app.inertia) return;
    const double decay = std::exp(-dt / app.inertiaTau);
    app.inertiaAccX += app.velX * dt;
    app.inertiaAccY += app.velY * dt;
    app.velX *= decay;
    app.velY *= decay;
    const int dx = (int)app.inertiaAccX, dy = (int)app.inertiaAccY;
    app.inertiaAccX -= dx;
    app.inertiaAccY -= dy;
    panPixels(app, dx, dy);
    if (std::hypot(app.velX, app.velY) < 15.0) app.inertia = false;
}

// Move the shown view toward the render view.
void tweenShown(App& app, double dt) {
    if (!app.tweening) return;
    const double k = 1.0 - std::exp(-dt / app.tweenTau);
    double logRatio = std::log(app.render.scale / app.shown.scale);
    app.shown.scale *= std::exp(logRatio * k);
    // Zooming in: the render pass computes a 1/8 margin around its view, so
    // the shown view may lag by at most that much or its edges have no data.
    if (app.shown.scale > app.render.scale * 1.2) app.shown.scale = app.render.scale * 1.2;
    logRatio = std::log(app.render.scale / app.shown.scale);
    // Centre follows from the anchor so the cursor point never drifts.
    app.shown.cx = app.anchorX;
    app.shown.cy = app.anchorY;
    app.shown.cx.subDouble(app.anchorPxX * app.shown.scale);
    app.shown.cy.addDouble(app.anchorPxY * app.shown.scale);
    if (std::fabs(logRatio) < 2e-4) snapShown(app);
}

// Half the image diagonal in complex units.
double halfDiagonal(const App& app) {
    return 0.5 * std::hypot((double)app.view.width, (double)app.view.height) * app.view.scale;
}

// View centre relative to the reference centre, into the view params.
// The view the field is currently laid out for: the newest generation's.
const View& fieldView(const App& app) { return app.gens.empty() ? app.render : app.gens[0].view; }

void updateRefOffset(App& app) {
    app.view.refOffX = BigFloat::diff(fieldView(app).cx, app.refCx);
    app.view.refOffY = BigFloat::diff(fieldView(app).cy, app.refCy);
    app.view.refRe = app.refCx.toDouble();
    app.view.refIm = app.refCy.toDouble();
    // A double c is exact to ~1e-16; keep the test well clear of that.
    app.view.interiorCheck = app.view.scale > 1e-12 ? 1 : 0;
}

void rebuildBla(App& app) {
    const double cMax = halfDiagonal(app) + std::hypot(app.view.refOffX, app.view.refOffY);
    buildBla(app.ref, cMax, std::exp2((double)app.blaEpsLog2), app.bla);
    buildBlaFloat(app.bla, app.blaF);
    app.renderer.uploadBla(app.bla.nodes.data(), app.blaF.data(), (int)app.bla.nodes.size(),
                           app.bla.levelOffset.data(), app.bla.levels, app.bla.steps);
}

// Move the reference to a point in the view (field pixel coordinates) that
// survives longer than the current one, and restart the affected pixels.
void rereferenceAt(App& app, int fx, int fy) {
    const double sc = app.view.scale;
    const double offX = ((double)(fx - app.renderer.marginX()) - 0.5 * app.view.width) * sc;
    const double offY = -(((double)(fy - app.renderer.marginY()) - 0.5 * app.view.height) * sc);
    app.refCx = fieldView(app).cx;
    app.refCy = fieldView(app).cy;
    app.refCx.addDouble(offX);
    app.refCy.addDouble(offY);
    updateRefOffset(app);
    computeReference(app.refCx, app.refCy, app.view.maxIter, 65536.0, app.ref);
    app.renderer.uploadReference(app.ref.zr.data(), app.ref.zi.data(), app.ref.length,
                                 app.ref.escaped);
    rebuildBla(app);
    app.renderer.restartPending(app.view);
    ++app.rerefRounds;
    app.rerefCheckedStride = 0;
}

void rebuildReference(App& app) {
    app.refCx = app.render.cx;
    app.refCy = app.render.cy;
    app.rerefRounds = 0;
    app.rerefCheckedStride = 0;
    updateRefOffset(app);
    computeReference(app.refCx, app.refCy, app.view.maxIter, 65536.0, app.ref);
    app.renderer.uploadReference(app.ref.zr.data(), app.ref.zi.data(), app.ref.length,
                                 app.ref.escaped);
    rebuildBla(app);
}

// Parameter groups for the settled display path.
bool paletteEqual(const ShadeParams& a, const ShadeParams& b) {
    if (a.paletteType != b.paletteType || a.stopCount != b.stopCount) return false;
    for (int i = 0; i < 3; ++i)
        if (a.a[i] != b.a[i] || a.b[i] != b.b[i] || a.c[i] != b.c[i] || a.d[i] != b.d[i]) return false;
    for (int i = 0; i < MAX_STOPS; ++i) {
        if (a.stopPos[i] != b.stopPos[i]) return false;
        for (int k = 0; k < 3; ++k)
            if (a.stopColor[i][k] != b.stopColor[i][k]) return false;
    }
    return true;
}

// Everything prepareSample() reads.
bool staticEqual(const ShadeParams& a, const ShadeParams& b) {
    return a.mode == b.mode && a.density == b.density && a.offset == b.offset &&
           a.transfer == b.transfer && a.anchor == b.anchor && a.iterBase == b.iterBase &&
           a.special == b.special && a.deStrength == b.deStrength &&
           a.lines == b.lines && a.lineDensity == b.lineDensity && a.lineWidth == b.lineWidth;
}

// Everything colourInput() reads besides time.
bool dynamicEqual(const ShadeParams& a, const ShadeParams& b) {
    for (int i = 0; i < 3; ++i)
        if (a.inside[i] != b.inside[i] || a.lineColor[i] != b.lineColor[i]) return false;
    return a.mode == b.mode && a.special == b.special && a.cycleSpeed == b.cycleSpeed &&
           a.waveSpeed == b.waveSpeed && a.lightSpeed == b.lightSpeed && a.slopes == b.slopes &&
           a.slopeAngle == b.slopeAngle && a.slopeHeight == b.slopeHeight &&
           a.slopeStrength == b.slopeStrength && a.lineStrength == b.lineStrength &&
           a.exposure == b.exposure;
}

// Mapping of a source view onto the shown view: where the shown centre
// lands in source pixels, and source pixels per shown pixel.
// sub: source pixels per display pixel at identity (ss for the render
// field, 1 for the last frame); srcW/H: source size in its own pixels.
void mapOnto(const App& app, const View& src, int sub, int srcW, int srcH, double& ox,
             double& oy, double& ratio) {
    const double srcScale = src.scale / sub;  // complex units per source pixel
    ratio = app.shown.scale / srcScale;
    ox = 0.5 * srcW + BigFloat::diff(app.shown.cx, src.cx) / srcScale;
    oy = 0.5 * srcH - BigFloat::diff(app.shown.cy, src.cy) / srcScale;
}

// Shading presets. Each fully replaces the shading parameters.
// Apply a preset (shading, post, animation flag).
void applyPresetTo(App& app, const Preset& p) {
    app.shade = p.shade;
    app.post = p.post;
    app.animate = p.animate;
}

void applyPreset(App& app, int which) { applyPresetTo(app, builtinPreset(which)); }

Preset currentLook(const App& app);

// Screenshot file stem: Pictures/mandelgpu/mandel_<stamp>_<zoom>.
std::string screenshotStem(App& app, int w) {
    const char* picsC = SDL_GetUserFolder(SDL_FOLDER_PICTURES);  // owned by SDL
    std::string dir = picsC ? picsC : "";
    dir += "Mandelbloom";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::time_t t = std::time(nullptr);
    char stamp[32];
    std::tm tmv{};
    localtime_s(&tmv, &t);
    std::strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", &tmv);
    const double zoom = 3.2 / (app.render.scale * w);
    char base[128];
    std::snprintf(base, sizeof base, "mandel_%s_%.2e", stamp, zoom);
    return dir + "/" + base;
}

// Linear (1.0 = SDR white) -> 8-bit sRGB with highlights rolled off softly.
void linearToSdrPng(const float* lin, int w, int h, std::vector<uint32_t>& px) {
    const float knee = 0.8f;
    auto roll = [&](float v) {
        if (v <= knee) return v;
        const float x = (v - knee) / (1.f - knee);
        return knee + (1.f - knee) * (x / (1.f + x));
    };
    auto enc = [](float v) {
        v = std::min(std::max(v, 0.f), 1.f);
        v = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
        return (uint32_t)(v * 255.f + 0.5f);
    };
    px.resize((size_t)w * h);
    for (size_t i = 0; i < px.size(); ++i) {
        const float* c = lin + i * 3;
        px[i] = enc(roll(c[0])) | (enc(roll(c[1])) << 8) | (enc(roll(c[2])) << 16) | 0xFF000000u;
    }
}

void writeLocationFile(App& app, const std::string& stem) {
    const int dg = std::max(5, (int)std::ceil(-std::log10(app.render.scale)) + 3);
    std::ofstream loc(stem + ".txt");
    // A complete command line for this exact picture.
    loc << app.render.cx.toString(dg) << " " << app.render.cy.toString(dg) << " "
        << app.render.scale << " " << app.view.maxIter << " --ss " << app.ss << " --aa "
        << app.aa.pattern << "," << app.aa.filter << "," << app.aa.radius << " --loadfile \""
        << stem << ".preset\"\n";
    savePresetFile(stem + ".preset", currentLook(app));
}

// Ctrl+F2: the presented frame including the UI, read back from the D3D12
// back buffer (scRGB), divided by SDR white and saved like F2.
void takeUiScreenshot(App& app) {
    std::vector<uint16_t> raw;
    int pitchPx = 0;
    if (!app.display.takeCapture(raw, pitchPx)) {
        app.toast = "screenshot failed";
        app.toastUntil = nowSeconds() + 3.0;
        return;
    }
    const int w = app.display.width(), h = app.display.height();
    auto halfToFloat = [](uint16_t v) {
        const uint32_t sgn = (v >> 15) & 1u, exp = (v >> 10) & 0x1Fu, man = v & 0x3FFu;
        float f;
        if (exp == 0) f = std::ldexp((float)man, -24);
        else if (exp == 31) f = man ? 0.f : 1e30f;
        else f = std::ldexp((float)(man | 0x400u), (int)exp - 25);
        return sgn ? -f : f;
    };
    const float inv = app.sdrWhite > 0.f ? 1.f / app.sdrWhite : 1.f;
    std::vector<float> lin((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const uint16_t* p = raw.data() + ((size_t)y * pitchPx + x) * 4;
            float* o = lin.data() + ((size_t)y * w + x) * 3;
            o[0] = halfToFloat(p[0]) * inv;
            o[1] = halfToFloat(p[1]) * inv;
            o[2] = halfToFloat(p[2]) * inv;
        }
    const std::string stem = screenshotStem(app, w) + "_ui";
    std::vector<uint32_t> px;
    linearToSdrPng(lin.data(), w, h, px);
    const bool ok = writePng((stem + ".png").c_str(), px.data(), w, h);
    writeLocationFile(app, stem);
    app.lastShotPath = stem;
    app.toast = ok ? "saved " + stem.substr(stem.find_last_of('/') + 1) + ".png" : "screenshot failed";
    app.toastUntil = nowSeconds() + 4.0;
    std::printf("screenshot %s ok=%d\n", stem.c_str(), ok ? 1 : 0);
}

// Save the current output. PNG: SDR, sRGB, highlights above SDR white
// rolled off softly. EXR: linear, 1.0 = SDR white, headroom preserved.
void takeScreenshot(App& app, bool alsoExr) {
    std::vector<float> lin;
    if (!app.renderer.readOutputLinear(lin)) {
        app.toast = "screenshot failed";
        app.toastUntil = nowSeconds() + 3.0;
        return;
    }
    const int w = app.view.width / app.ss, h = app.view.height / app.ss;
    const std::string stem = screenshotStem(app, w);
    const std::string base = stem.substr(stem.find_last_of('/') + 1);
    std::vector<uint32_t> px;
    linearToSdrPng(lin.data(), w, h, px);
    bool ok = writePng((stem + ".png").c_str(), px.data(), w, h);
    if (alsoExr) ok = writeExr((stem + ".exr").c_str(), lin.data(), w, h) && ok;
    writeLocationFile(app, stem);
    app.lastShotPath = stem;
    app.toast = ok ? std::string("saved ") + base + (alsoExr ? ".png + .exr" : ".png")
                   : "screenshot failed";
    app.toastUntil = nowSeconds() + 4.0;
    std::printf("screenshot %s ok=%d\n", stem.c_str(), ok ? 1 : 0);
}

// Freeze the animation where it is: fold the accumulated phase into the
// visible parameters so the UI (and saved presets) describe the picture.
void setAnimate(App& app, bool on) {
    if (app.animate && !on) {
        ShadeParams& sp = app.shade;
        const float t = app.animTime;
        sp.offset = sp.offset + sp.cycleSpeed * t;
        sp.offset -= std::floor(sp.offset);
        sp.slopeAngle = std::fmod(sp.slopeAngle + sp.lightSpeed * t + 360.f * 1000.f, 360.f);
        sp.wavePhase = sp.wavePhase + sp.waveSpeed * t;
        sp.wavePhase -= std::floor(sp.wavePhase);
        app.animTime = 0.f;
    }
    app.animate = on;
}

// Sidecar preset describing the current look exactly.
Preset currentLook(const App& app) {
    Preset cur;
    cur.shade = app.shade;
    cur.post = app.post;
    cur.animate = app.animate;
    if (app.animate) {
        // Bake the phase at the moment of capture.
        const float t = app.animTime;
        cur.shade.offset -= std::floor(cur.shade.offset + cur.shade.cycleSpeed * t);
        cur.shade.offset += cur.shade.cycleSpeed * t;
        cur.shade.slopeAngle = std::fmod(cur.shade.slopeAngle + cur.shade.lightSpeed * t + 360.f * 1000.f, 360.f);
        cur.shade.wavePhase += cur.shade.waveSpeed * t;
        cur.shade.wavePhase -= std::floor(cur.shade.wavePhase);
        cur.animate = false;
    }
    return cur;
}

void setFullscreen(App& app, bool on) {
    app.fullscreen = on;
    SDL_SetWindowFullscreen(app.window, on);
}

bool handleEvent(App& app, const SDL_Event& e) {
    ImGui_ImplSDL3_ProcessEvent(&e);
    const ImGuiIO& io = ImGui::GetIO();
    switch (e.type) {
        case SDL_EVENT_QUIT:
            return false;
        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
            app.uiScale = 0.f;  // force re-apply
            applyUiScale(app);
            break;
        case SDL_EVENT_WINDOW_HDR_STATE_CHANGED:
            readHdrState(app);
            app.lastPostValid = false;  // output scale changed: re-run post
            break;
        case SDL_EVENT_KEY_DOWN:
            if (io.WantCaptureKeyboard) break;
            if (e.key.key == SDLK_ESCAPE) {
                if (app.video.active) {
                    app.video.cancel = true;
                    break;
                }
                return false;
            }
            if (e.key.key == SDLK_R) resetView(app);
            if (e.key.key == SDLK_TAB) app.showUi = !app.showUi;
            if (e.key.key == SDLK_F11) setFullscreen(app, !app.fullscreen);
            if (e.key.key == SDLK_F2)
                app.screenshotRequest = (e.key.mod & SDL_KMOD_CTRL) ? 3
                                        : (e.key.mod & SDL_KMOD_SHIFT) ? 2 : 1;
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (io.WantCaptureMouse) break;
            if (e.button.button == SDL_BUTTON_LEFT) {
                app.dragging = true;
                app.inertia = false;  // grabbing stops the coast
                app.velX = app.velY = 0;
                app.lastMotionSec = nowSeconds();
            }
            break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (e.button.button == SDL_BUTTON_LEFT && app.dragging) {
                app.dragging = false;
                // Release after a pause means "stop here", not "fling".
                const bool fresh = nowSeconds() - app.lastMotionSec < 0.08;
                if (app.inertiaEnabled && fresh && std::hypot(app.velX, app.velY) > 50.0) {
                    app.inertia = true;
                    app.inertiaAccX = app.inertiaAccY = 0;
                } else {
                    app.velX = app.velY = 0;
                }
            }
            break;
        case SDL_EVENT_MOUSE_MOTION:
            if (app.dragging && !io.WantCaptureMouse) {
                const int dx = (int)e.motion.xrel, dy = (int)e.motion.yrel;
                const double t = nowSeconds();
                const double dt = std::max(t - app.lastMotionSec, 0.008);
                app.lastMotionSec = t;
                // Smoothed velocity from the last few motion events.
                const double k = dt > 0.05 ? 1.0 : 0.4;
                app.velX += (dx / dt - app.velX) * k;
                app.velY += (dy / dt - app.velY) * k;
                panPixels(app, dx, dy);
            }
            break;
        case SDL_EVENT_MOUSE_WHEEL: {
            if (io.WantCaptureMouse) break;
            app.inertia = false;
            const double factor = std::pow(0.8, (double)e.wheel.y);
            zoomAt(app, e.wheel.mouse_x, e.wheel.mouse_y, factor);
            break;
        }
        default:
            break;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Video export

std::string videoStem(double zoom) {
    const char* vidsC = SDL_GetUserFolder(SDL_FOLDER_VIDEOS);  // owned by SDL
    std::string dir = vidsC ? vidsC : "";
    dir += "Mandelbloom";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::time_t t = std::time(nullptr);
    char stamp[32];
    std::tm tmv{};
    localtime_s(&tmv, &t);
    std::strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", &tmv);
    char base[128];
    std::snprintf(base, sizeof base, "mandel_%s_%.2e", stamp, zoom);
    return dir + "/" + base;
}

// Scale (complex units per output pixel) of output frame f.
double videoFrameScale(const VideoJob& job, int f) {
    const VideoParams& p = job.p;
    const double t = f / p.fps;
    double u = p.seconds > 0 ? (t - p.holdStart) / p.seconds : 1.0;
    u = std::min(std::max(u, 0.0), 1.0);
    const double sm = u * u * (3.0 - 2.0 * u);
    u += (sm - u) * p.ease;
    return job.startScale * std::pow(job.endScale / job.startScale, u);
}

// The coarser of the two keyframes bracketing scale s.
int videoOuter(const VideoJob& job, double s) {
    const int k = (int)std::floor(std::log2(job.startScale / s) + 1e-9);
    return std::min(std::max(k, 0), job.keyCount - 1);
}

View videoKeyView(const App& app, int k) {
    View v = app.video.target;
    v.scale = app.video.startScale / std::ldexp(1.0, k);
    return v;
}

void videoStartKey(App& app, int k) {
    VideoJob& job = app.video;
    const View kv = videoKeyView(app, k);
    app.render = kv;
    syncViewScale(app);
    // Iterations needed grow roughly linearly with depth; shallow keyframes
    // would otherwise spend the full budget on every interior pixel.
    if (job.p.iterRamp && job.keyCount > 1) {
        const int full = job.savedView.maxIter;
        const int lo = std::min(full, 2000);
        const double f = std::min(1.0, 2.0 * k / (double)(job.keyCount - 1));
        app.view.maxIter = (int)std::lround(lo + (full - lo) * f);
    }
    App::Gen g;
    g.view = kv;
    g.id = app.nextGen;
    app.nextGen = app.nextGen % 65535 + 1;
    app.gens.clear();
    app.gens.push_back(g);
    // A re-reference during the previous keyframe moved the orbit off the
    // target; bring it back (as the interactive path does per generation).
    if (app.rerefRounds > 0) {
        rebuildReference(app);
    } else {
        app.rerefCheckedStride = 0;
        updateRefOffset(app);
        rebuildBla(app);
    }
    app.renderer.beginIterate(app.view, g.id);
    VideoJob::Key& key = job.keys[k];
    key.gen = g.id;
    key.slot = 1 + (k % 2);
    key.ready = false;
    key.minIter = -1.f;
    job.iterating = k;
}

void videoFinish(App& app, bool cancelled) {
    VideoJob& job = app.video;
    // A cancel still closes the stream cleanly so the frames written so far
    // play; only a broken pipe kills ffmpeg.
    int code = 0;
    if (cancelled && !job.error.empty()) job.enc.cancel();
    else code = job.enc.finish();
    app.renderer.endVideo();
    app.render = job.savedRender;
    app.shown = job.savedShown;
    app.view = job.savedView;
    app.gens.clear();
    app.dirty = true;
    app.ssApplied = 0;  // re-bind the display next frame
    app.renderer.setOutputScale(app.sdrWhite, app.hdrHeadroom);
    app.lastShadeValid = false;
    app.lastPostValid = false;
    app.cacheValid = false;
    job.active = false;
    const double secs = nowSeconds() - job.t0;
    char msg[512];
    if (cancelled && !job.error.empty())
        std::snprintf(msg, sizeof msg, "video failed: %s", job.error.c_str());
    else if (cancelled)
        std::snprintf(msg, sizeof msg, "video cancelled, kept %d frames in %s", job.frame,
                      job.outPath.c_str());
    else if (code != 0)
        std::snprintf(msg, sizeof msg, "ffmpeg exited with %d (see .ffmpeg.log)", code);
    else
        std::snprintf(msg, sizeof msg, "saved %s (%d frames, %.0f s)", job.outPath.c_str(),
                      job.frame, secs);
    app.toast = msg;
    app.toastUntil = nowSeconds() + 10.0;
    app.lastVideoPath = job.outPath;
    std::printf("video %s frames=%d/%d keyframes=%d/%d seconds=%.1f keyMs=%.0f code=%d%s%s\n",
                job.outPath.c_str(), job.frame, job.totalFrames, job.keysDone, job.keyCount, secs,
                job.keyMs, code, cancelled ? " cancelled" : "", job.error.empty() ? "" : " error");
    std::fflush(stdout);
}

bool startVideo(App& app) {
    VideoJob& job = app.video;
    if (job.active) return false;
    auto fail = [&](const std::string& why) {
        app.toast = why;
        app.toastUntil = nowSeconds() + 8.0;
        std::printf("video not started: %s\n", why.c_str());
        std::fflush(stdout);
        return false;
    };
    const VideoParams p = app.videoParams;
    const int winW = app.view.width / app.ss, winH = app.view.height / app.ss;
    int w = p.width > 0 ? p.width : winW, h = p.height > 0 ? p.height : winH;
    w &= ~1;
    h &= ~1;
    if (w < 16 || h < 16 || p.fps <= 0) return fail("video: bad size or fps");
    job.p = p;
    job.w = w;
    job.h = h;
    job.nits = p.nits > 0.f ? p.nits : (app.hdrEnabled ? 80.f * app.sdrWhite : 203.f);
    // Tone map into the display's headroom for both: the HDR video carries
    // it, the SDR video rolls it off like an F2 screenshot.
    job.headroom = p.headroom > 0.f ? p.headroom : std::max(app.hdrHeadroom, 1.f);
    // Same framing as the window at the target; keyframes 2x apart in zoom.
    job.target = app.render;
    job.target.scale = app.render.scale * ((double)winW / w);
    job.endScale = job.target.scale;
    job.startScale = 3.2 / (std::max(p.zoomFrom, 1e-6) * w);
    if (job.startScale <= job.endScale * 1.0001) return fail("video: start zoom is past the target");
    job.keyCount = (int)std::ceil(std::log2(job.startScale / job.endScale) - 1e-9) + 1;
    job.keys.assign((size_t)job.keyCount, VideoJob::Key{});
    job.keysDone = 0;
    job.totalFrames = std::max(1, (int)std::llround((p.holdStart + p.seconds + p.holdEnd) * p.fps));
    job.frame = 0;
    job.nextKey = 0;
    job.iterating = -1;
    job.keyMs = 0;
    job.error.clear();
    job.cancel = false;
    job.previewOk = true;

    const double zoom = 3.2 / (job.endScale * w);
    std::string stem = p.outPath;
    if (stem.empty()) {
        stem = videoStem(zoom);
    } else if (stem.size() > 4 && stem.compare(stem.size() - 4, 4, ".mp4") == 0) {
        stem.resize(stem.size() - 4);
    }
    job.outPath = stem + ".mp4";
    writeLocationFile(app, stem);  // command line + .preset sidecar, like F2
    {
        std::ofstream loc(stem + ".txt", std::ios::app);
        loc << "--video-seconds " << p.seconds << " --video-fps " << p.fps << " --video-size " << w
            << "x" << h << " --video-from " << p.zoomFrom << (p.hdr ? "" : " --video-sdr")
            << " --video-cq " << p.cq << " --video-hold " << p.holdStart << "," << p.holdEnd
            << " --video-ease " << p.ease << " --video-nits " << job.nits << "\n";
    }
    VideoEncodeSettings es;
    es.outPath = job.outPath;
    es.width = w;
    es.height = h;
    es.fps = p.fps;
    es.hdr = p.hdr;
    es.cq = p.cq;
    es.codec = p.codec;
    es.logPath = stem + ".ffmpeg.log";
    if (!job.enc.start(es)) return fail("ffmpeg failed to start: " + job.enc.error());

    // Take over the renderer.
    job.savedRender = app.render;
    job.savedShown = app.shown;
    job.savedView = app.view;
    if (!app.renderer.bindVideo(w, h, app.ss, 2)) {
        job.enc.cancel();
        return fail("video: could not allocate the render buffers");
    }
    app.view.width = w * app.ss;
    app.view.height = h * app.ss;
    app.render = job.target;
    app.shown = job.target;
    syncViewScale(app);
    app.gens.clear();
    rebuildReference(app);  // once, at the target; keyframes only rebuild the BLA table
    app.renderer.buildPaletteLut(app.shade);
    app.renderer.setOutputScale(1.f, job.headroom);
    job.t0 = nowSeconds();
    job.active = true;
    std::printf("video start %s %dx%d fps=%g frames=%d keyframes=%d hdr=%d nits=%.0f headroom=%.2f\n",
                job.outPath.c_str(), w, h, p.fps, job.totalFrames, job.keyCount, p.hdr ? 1 : 0,
                job.nits, job.headroom);
    std::printf("  ffmpeg: %s\n", job.enc.commandLine().c_str());
    std::fflush(stdout);
    app.toast = "rendering " + job.outPath;
    app.toastUntil = nowSeconds() + 4.0;
    return true;
}

// Debug: write keyframe k as a PNG, composited 1:1 from its slot.
void videoDumpKey(App& app, int k, const char* suffix) {
    VideoJob& job = app.video;
    const VideoJob::Key& key = job.keys[(size_t)k];
    const View kv = videoKeyView(app, k);
    const double savedScale = app.shown.scale;
    app.shown.scale = kv.scale;
    CompositeMap m;
    m.ss = 2;
    m.genCount = 1;
    GenMap& gm = m.gens[0];
    mapOnto(app, kv, app.ss, app.view.width, app.view.height, gm.ox, gm.oy, gm.ratio);
    gm.pixelScale = (float)(kv.scale / app.ss);
    gm.gen = key.gen;
    gm.slot = key.slot;
    app.renderer.composite(app.shade, 0.f, m);
    app.renderer.postProcess(app.post, 0);
    std::vector<uint32_t> px;
    if (app.renderer.readOutput(px)) {
        char name[96];
        std::snprintf(name, sizeof name, "_k%03d%s.png", k, suffix);
        const std::string stem = job.outPath.substr(0, job.outPath.size() - 4);
        writePng((stem + name).c_str(), px.data(), job.w, job.h);
    }
    app.shown.scale = savedScale;
}

// Keyframe pipeline: keep the live field iterating the next keyframe whose
// slot is free, stash finished ones.
void videoPumpKeys(App& app) {
    VideoJob& job = app.video;
    if (app.renderer.iterateBusy()) return;
    if (job.iterating >= 0) {
        const bool sliceDone = app.renderer.takeSliceFinished();
        const int done = app.renderer.completedStride();
        if (sliceDone && app.ref.escaped && done > 0 && done != app.rerefCheckedStride &&
            app.rerefRounds < 12) {
            app.rerefCheckedStride = done;
            int fx, fy;
            if (app.renderer.findUnreliable(fx, fy)) {
                rereferenceAt(app, fx, fy);
                std::printf("video keyframe %d: re-reference %d at stride %d\n", job.iterating,
                            app.rerefRounds, done);
            }
        }
        if (!app.renderer.iterateDone()) {
            app.renderer.stepIterate();
            ++job.slices;
            return;
        }
        VideoJob::Key& key = job.keys[(size_t)job.iterating];
        if (job.p.dumpKeys) {
            CudaRenderer::DebugStats st;
            if (app.renderer.debugStats(key.gen, st))
                std::printf("video keyframe %d gen %d: total=%d match=%d zero=%d other=%d active=%d "
                            "escaped=%d inside=%d matchInside=%d iterMs=%.0f min=%.0f\n",
                            job.iterating, key.gen, st.total, st.genMatch, st.genZero, st.genOther,
                            st.stActive, st.stEscaped, st.stInside, st.genMatchInside,
                            app.renderer.lastIterateMs(), app.renderer.minIter());
            std::fflush(stdout);
        }
        app.renderer.stashField(key.slot);
        if (job.p.dumpKeys) videoDumpKey(app, job.iterating, "");
        key.minIter = app.renderer.minIter();
        key.ready = true;
        job.keyMs += app.renderer.lastIterateMs();
        job.lastKeySec = app.renderer.lastIterateMs() / 1000.0;
        job.iterating = -1;
        ++job.keysDone;
    }
    if (job.nextKey < job.keyCount) {
        const int k = job.nextKey;
        const int outer = videoOuter(job, videoFrameScale(job, job.frame));
        if (k < 2 || k - 2 < outer) {  // its slot's previous keyframe is no longer needed
            videoStartKey(app, k);
            ++job.nextKey;
        }
    }
}

void videoStep(App& app) {
    VideoJob& job = app.video;
    if (job.cancel) {
        videoFinish(app, true);
        return;
    }
    double tA = nowSeconds();
    videoPumpKeys(app);
    double tB = nowSeconds();
    job.tPump += tB - tA;
    const double budgetEnd = tB + 0.05;  // then let the window update
    while (job.frame < job.totalFrames && nowSeconds() < budgetEnd) {
        const double s = videoFrameScale(job, job.frame);
        const int outer = videoOuter(job, s);
        const int inner = outer + 1 < job.keyCount ? outer + 1 : -1;
        if (!job.keys[(size_t)outer].ready || (inner >= 0 && !job.keys[(size_t)inner].ready)) break;
        app.shown.scale = s;
        CompositeMap m;
        m.ss = std::max(1, job.p.subsamples);
        m.genCount = 0;
        auto add = [&](int k) {
            GenMap& gm = m.gens[m.genCount++];
            const View kv = videoKeyView(app, k);
            mapOnto(app, kv, app.ss, app.view.width, app.view.height, gm.ox, gm.oy, gm.ratio);
            gm.pixelScale = (float)(kv.scale / app.ss);
            gm.gen = job.keys[(size_t)k].gen;
            gm.slot = job.keys[(size_t)k].slot;
        };
        if (inner >= 0) add(inner);
        add(outer);
        // Palette anchor: between the two keyframes' minima.
        float base = 0.f;
        if (app.shade.anchor) {
            const float mo = job.keys[(size_t)outer].minIter;
            const float mi = inner >= 0 ? job.keys[(size_t)inner].minIter : -1.f;
            if (mo >= 0.f && mi >= 0.f) {
                const double frac = std::log2(job.startScale / s) - outer;
                base = (float)(mo + (mi - mo) * std::min(std::max(frac, 0.0), 1.0));
            } else {
                base = std::max(mo, mi);
            }
            base = std::floor(std::max(base, 0.f));
        }
        app.shade.iterBase = base;
        const float t = app.animate ? (float)(job.frame / job.p.fps) : 0.f;
        app.renderer.composite(app.shade, t, m);
        app.renderer.postProcess(app.post, (uint32_t)job.frame);
        size_t bytes = 0;
        const void* buf = app.renderer.convertFrame(
            job.p.hdr ? CudaRenderer::VIDEO_P010 : CudaRenderer::VIDEO_YUV420P8, job.nits, bytes);
        if (!buf) {
            job.error = "frame conversion failed";
            job.cancel = true;
            break;
        }
        if (!job.enc.submit(buf, bytes)) {
            job.error = job.enc.error();
            job.cancel = true;
            break;
        }
        ++job.frame;
        ++job.framesSince;
        {
            const double now = nowSeconds();
            if (job.lastFrameStamp > 0) {
                const double dtf = now - job.lastFrameStamp;
                job.frameSec = job.frameSec > 0 ? 0.9 * job.frameSec + 0.1 * dtf : dtf;
            }
            job.lastFrameStamp = now;
        }
        videoPumpKeys(app);
    }
    tA = nowSeconds();
    job.tFrames += tA - tB;
    if (job.frame == 0 && job.iterating >= 0 && !app.gens.empty()) {
        // Nothing encoded yet: show the keyframe being computed, as the
        // interactive view would.
        app.shown.scale = app.render.scale;
        CompositeMap m;
        m.ss = app.ss;
        m.genCount = 1;
        GenMap& gm = m.gens[0];
        mapOnto(app, app.render, app.ss, app.view.width, app.view.height, gm.ox, gm.oy, gm.ratio);
        gm.pixelScale = (float)(app.render.scale / app.ss);
        gm.gen = app.gens[0].id;
        gm.slot = 0;
        app.renderer.composite(app.shade, 0.f, m);
        app.renderer.postProcess(app.post, 0);
    }
    tB = nowSeconds();
    job.tLive += tB - tA;
    if (job.previewOk) app.renderer.previewToDisplay(app.sdrWhite);
    app.renderer.syncDisplay();
    tA = nowSeconds();
    job.tPreview += tA - tB;
    if (job.p.debug && tA - job.tReport >= 2.0) {
        std::printf("video t=%.0fs frames=%d (+%d) keys=%d/%d iterating=%d slices=%d pump=%.2fs "
                    "frames=%.2fs live=%.2fs preview=%.2fs iterMs=%.0f stride=%d" "%s",
                    tA - job.t0, job.frame, job.framesSince, job.keysDone, job.keyCount,
                    job.iterating, job.slices, job.tPump, job.tFrames, job.tLive, job.tPreview,
                    app.renderer.lastIterateMs(), app.renderer.currentStride(), "\n");
        std::fflush(stdout);
        job.tReport = tA;
        job.tPump = job.tFrames = job.tLive = job.tPreview = 0;
        job.slices = 0;
        job.framesSince = 0;
    }
    if (!job.cancel && job.frame >= job.totalFrames) videoFinish(app, false);
}

void drawVideoUi(App& app) {
    if (!ImGui::CollapsingHeader("Video")) return;
    VideoJob& job = app.video;
    VideoParams& vp = app.videoParams;
    if (job.active) {
        const double el = nowSeconds() - job.t0;
        const float prog = job.totalFrames > 0 ? (float)job.frame / job.totalFrames : 0.f;
        const double eta = (job.keyCount - job.keysDone) * job.lastKeySec +
                           (job.totalFrames - job.frame) * job.frameSec;
        char label[96];
        std::snprintf(label, sizeof label, "%d / %d frames", job.frame, job.totalFrames);
        ImGui::ProgressBar(prog, ImVec2(-FLT_MIN, 0), label);
        ImGui::Text("keyframes %d / %d   elapsed %.0f s   eta %.0f s", job.keysDone, job.keyCount,
                    el, eta);
        if (job.iterating >= 0)
            ImGui::Text("computing keyframe %d (%d iterations max)", job.iterating, app.view.maxIter);
        ImGui::TextWrapped("%s", job.outPath.c_str());
        if (ImGui::Button("cancel (Esc)")) job.cancel = true;
        return;
    }
    const char* sizes[] = {"window", "1920 x 1080", "2560 x 1440", "3840 x 2160"};
    static const int sw[] = {0, 1920, 2560, 3840}, sh[] = {0, 1080, 1440, 2160};
    ImGui::Combo("size", &app.videoSizeChoice, sizes, 4);
    vp.width = sw[app.videoSizeChoice];
    vp.height = sh[app.videoSizeChoice];
    float secs = (float)vp.seconds;
    if (ImGui::SliderFloat("seconds", &secs, 2.f, 1200.f, "%.0f", ImGuiSliderFlags_Logarithmic))
        vp.seconds = secs;
    int fpsChoice = vp.fps >= 100 ? 2 : vp.fps >= 45 ? 1 : 0;
    const char* fpss[] = {"30", "60", "120"};
    if (ImGui::Combo("fps", &fpsChoice, fpss, 3)) vp.fps = fpsChoice == 2 ? 120.0 : fpsChoice == 1 ? 60.0 : 30.0;
    const double zoomEnd = 3.2 / (app.render.scale * std::max(1, app.view.width / app.ss));
    const float maxLog = (float)std::max(0.1, std::log10(std::max(zoomEnd, 1.0)) - 0.3);
    float fromLog = (float)std::log10(std::max(vp.zoomFrom, 1.0));
    if (ImGui::SliderFloat("start zoom", &fromLog, 0.f, maxLog, "1e%.1f")) vp.zoomFrom = std::pow(10.0, fromLog);
    ImGui::SliderFloat("ease", &vp.ease, 0.f, 1.f);
    float hs = (float)vp.holdStart, he = (float)vp.holdEnd;
    if (ImGui::SliderFloat("hold start", &hs, 0.f, 10.f, "%.1f s")) vp.holdStart = hs;
    if (ImGui::SliderFloat("hold end", &he, 0.f, 10.f, "%.1f s")) vp.holdEnd = he;
    ImGui::Checkbox("HDR (10-bit PQ)", &vp.hdr);
    ImGui::SameLine();
    ImGui::Checkbox("iteration ramp", &vp.iterRamp);
    ImGui::SliderInt("quality (cq)", &vp.cq, 10, 35);
    const double doublings = std::log2(std::max(zoomEnd, 1.0) / std::max(vp.zoomFrom, 1.0));
    ImGui::Text("%.0f doublings, %.2f per second", doublings, vp.seconds > 0 ? doublings / vp.seconds : 0.0);
    if (ImGui::Button("render video")) startVideo(app);
    if (!app.lastVideoPath.empty()) ImGui::TextWrapped("last: %s", app.lastVideoPath.c_str());
}

void drawUi(App& app) {
    if (!app.showUi) return;
    ImGui::SetNextWindowPos(ImVec2(10 * app.uiScale, 10 * app.uiScale), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.75f);
    if (ImGui::Begin("mandelbloom", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted(app.renderer.deviceName());
        ImGui::Separator();
        const int digits = std::max(5, (int)std::ceil(-std::log10(app.render.scale)) + 3);
        ImGui::Text("center  %s", app.render.cx.toString(digits).c_str());
        ImGui::Text("        %s i", app.render.cy.toString(digits).c_str());
        const double zoom = 3.2 / (app.render.scale * (app.view.width / app.ss));
        ImGui::Text("zoom    %.3g x   (%ld bits)", zoom, (long)app.render.cx.prec());
        ImGui::Text("size    %d x %d  (field %d x %d)", app.view.width / app.ss,
                    app.view.height / app.ss, app.view.width, app.view.height);
        {
            const char* items[] = {"1x", "2x", "3x"};
            int sel = app.ss - 1;
            ImGui::SetNextItemWidth(80 * app.uiScale);
            if (ImGui::Combo("supersample", &sel, items, 3)) app.ss = sel + 1;
            const char* pats[] = {"grid", "rotated", "stochastic"};
            ImGui::SetNextItemWidth(100 * app.uiScale);
            if (ImGui::Combo("samples", &app.aa.pattern, pats, AA_PATTERN_COUNT)) {
                app.view.aaPattern = app.aa.pattern;
                app.dirty = true;
            }
            const char* filts[] = {"box", "tent", "gaussian", "blackman"};
            ImGui::SetNextItemWidth(100 * app.uiScale);
            if (ImGui::Combo("filter", &app.aa.filter, filts, FILTER_COUNT)) app.cachedShown = false;
            if (app.aa.filter != FILTER_BOX) {
                ImGui::SetNextItemWidth(100 * app.uiScale);
                if (ImGui::SliderFloat("radius px", &app.aa.radius, 0.f, 3.f)) app.cachedShown = false;
            }
        }
        if (ImGui::SliderInt("max iter", &app.view.maxIter, 64, 65536, "%d",
                             ImGuiSliderFlags_Logarithmic)) {
            app.dirty = true;
        }
        ImGui::Separator();
        ImGui::Text("ref     %.2f ms  (%d iters%s, %d re-refs)", app.ref.computeMs, app.ref.length,
                    app.ref.escaped ? ", escaped" : "", app.rerefRounds);
        ImGui::Text("bla     %.2f ms  (%d levels, %zu nodes)", app.bla.buildMs, app.bla.levels,
                    app.bla.nodes.size());
        if (ImGui::Checkbox("use BLA", &app.view.useBla)) app.dirty = true;
        ImGui::SameLine();
        if (ImGui::Checkbox("float kernel", &app.view.useFloat)) app.dirty = true;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120 * app.uiScale);
        if (ImGui::SliderFloat("eps log2", &app.blaEpsLog2, -40.f, -8.f, "%.0f")) app.dirty = true;
        if (app.renderer.iterateDone()) {
            ImGui::Text("iterate %.0f ms", app.renderer.lastIterateMs());
        } else {
            const float pr = app.renderer.progress();
            const float eta = app.renderer.etaMs();
            char label[64];
            if (eta >= 0.f) std::snprintf(label, sizeof label, "%.0f%%  %.1f s left", pr * 100.f, eta / 1000.f);
            else std::snprintf(label, sizeof label, "%.0f%%", pr * 100.f);
            // Bar without ImGui's moving label, then the label centred and
            // outlined so it reads over both the filled and empty parts.
            ImGui::ProgressBar(pr, ImVec2(-1.f, 0.f), "");
            const ImVec2 mn = ImGui::GetItemRectMin(), mx = ImGui::GetItemRectMax();
            const ImVec2 ts = ImGui::CalcTextSize(label);
            const ImVec2 pos(0.5f * (mn.x + mx.x - ts.x), 0.5f * (mn.y + mx.y - ts.y));
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImU32 outline = IM_COL32(0, 0, 0, 255), fill = IM_COL32(255, 255, 255, 255);
            for (int oy = -1; oy <= 1; ++oy)
                for (int ox = -1; ox <= 1; ++ox)
                    if (ox || oy) dl->AddText(ImVec2(pos.x + ox, pos.y + oy), outline, label);
            dl->AddText(pos, fill, label);
            ImGui::Text("iterate %.0f ms  (stride %d, slice %.1f ms)", app.renderer.lastIterateMs(),
                        app.renderer.currentStride(), app.renderer.lastSliceMs());
        }
        ImGui::Text("display %.2f ms%s", app.renderer.lastShadeMs(),
                    app.cachedShown ? "  (cached)" : "");
        ImGui::Text("post    %.2f ms", app.renderer.lastPostMs());
        ImGui::Text("output  %s  SDR white %.2f  headroom %.2f",
                    app.hdrEnabled ? "HDR scRGB FP16" : "SDR (FP16 scRGB)", app.sdrWhite,
                    app.hdrHeadroom);
        ImGui::Checkbox("vsync", &app.vsync);
        ImGui::SameLine();
        bool fs = app.fullscreen;
        if (ImGui::Checkbox("fullscreen (F11)", &fs)) setFullscreen(app, fs);
        ImGui::Text("frame   %.0f fps", app.fps);
        ImGui::Checkbox("smooth zoom", &app.tweenEnabled);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100 * app.uiScale);
        ImGui::SliderFloat("##tau", &app.tweenTau, 0.02f, 0.3f, "%.2f s");
        ImGui::Checkbox("pan inertia", &app.inertiaEnabled);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100 * app.uiScale);
        ImGui::SliderFloat("##itau", &app.inertiaTau, 0.05f, 1.0f, "%.2f s");
        ImGui::TextDisabled("drag: pan  wheel: zoom  R: reset  Tab: hide  F11: fullscreen");
        ImGui::TextDisabled("F2: screenshot   Shift+F2: + EXR   Ctrl+F2: with UI");

        if (ImGui::CollapsingHeader("Shading")) {
            ShadeParams& sp = app.shade;
            ImGui::TextDisabled("presets");
            for (int i = 0; i < BUILTIN_PRESET_COUNT; ++i) {
                if (i % 4 != 0) ImGui::SameLine();
                ImGui::PushID(1000 + i);  // "angle" and "distance" also name sliders
                if (ImGui::SmallButton(kBuiltinPresetNames[i])) applyPreset(app, i);
                ImGui::PopID();
            }
            if (!app.savedPresetsValid) {
                app.savedPresets = listSavedPresets();
                app.savedPresetsValid = true;
            }
            ImGui::SetNextItemWidth(160 * app.uiScale);
            ImGui::InputText("##pname", app.presetName, sizeof app.presetName);
            ImGui::SameLine();
            if (ImGui::SmallButton("save")) {
                Preset cur;
                cur.shade = app.shade;
                cur.post = app.post;
                cur.animate = app.animate;
                if (savePreset(app.presetName, cur)) app.savedPresetsValid = false;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(F2 also writes a .preset beside the shot)");
            for (size_t i = 0; i < app.savedPresets.size(); ++i) {
                ImGui::PushID((int)i);
                if (ImGui::SmallButton(app.savedPresets[i].c_str())) {
                    Preset pr;
                    if (loadPreset(app.savedPresets[i], pr)) {
                        applyPresetTo(app, pr);
                        std::snprintf(app.presetName, sizeof app.presetName, "%s",
                                      app.savedPresets[i].c_str());
                    }
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("x")) {
                    deletePreset(app.savedPresets[i]);
                    app.savedPresetsValid = false;
                }
                ImGui::PopID();
            }
            ImGui::TextDisabled("%s", presetDir().c_str());
            const char* modes[] = {"smooth", "log steps", "wave", "panels", "angle", "distance"};
            ImGui::Combo("mode", &sp.mode, modes, SHADE_MODE_COUNT);
            if (sp.mode == SHADE_LOGSTEPS || sp.mode == SHADE_WAVE || sp.mode == SHADE_PANELS) {
                ImGui::SliderFloat("per cycle", &sp.special, 1.f, 16.f, "%.1f");
            } else if (sp.mode == SHADE_ANGLE) {
                ImGui::SliderFloat("cycles per turn", &sp.special, 0.25f, 8.f, "%.2f");
            } else if (sp.mode == SHADE_DISTANCE) {
                ImGui::SliderFloat("decades per cycle", &sp.special, 0.5f, 8.f, "%.1f");
            }
            ImGui::Combo("transfer", &sp.transfer, "linear\0sqrt\0log\0");
            ImGui::SliderFloat("density", &sp.density, 4.f, 16384.f, "%.1f",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("offset", &sp.offset, 0.f, 1.f);
            {
                bool anchor = sp.anchor != 0;
                if (ImGui::Checkbox("anchor to view minimum", &anchor)) sp.anchor = anchor;
                if (sp.anchor) {
                    ImGui::SameLine();
                    ImGui::TextDisabled("(%.0f iters)", sp.iterBase);
                }
            }

            ImGui::SeparatorText("palette");
            ImGui::RadioButton("gradient", &sp.paletteType, 0);
            ImGui::SameLine();
            ImGui::RadioButton("cosine", &sp.paletteType, 1);
            if (sp.paletteType == 0) {
                for (int i = 0; i < sp.stopCount; ++i) {
                    ImGui::PushID(i);
                    ImGui::ColorEdit3("##c", sp.stopColor[i],
                                      ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(140 * app.uiScale);
                    const float lo = i == 0 ? 0.f : sp.stopPos[i - 1];
                    const float hi = i == sp.stopCount - 1 ? 1.f : sp.stopPos[i + 1];
                    if (i == 0 || i == sp.stopCount - 1) {
                        ImGui::TextDisabled(i == 0 ? "start" : "end");
                    } else {
                        ImGui::SliderFloat("##p", &sp.stopPos[i], lo, hi, "%.2f");
                    }
                    if (sp.stopCount > 2 && i > 0 && i < sp.stopCount - 1) {
                        ImGui::SameLine();
                        if (ImGui::SmallButton("x")) {
                            for (int k = i; k + 1 < sp.stopCount; ++k) {
                                sp.stopPos[k] = sp.stopPos[k + 1];
                                for (int c = 0; c < 3; ++c) sp.stopColor[k][c] = sp.stopColor[k + 1][c];
                            }
                            --sp.stopCount;
                        }
                    }
                    ImGui::PopID();
                }
                if (sp.stopCount < MAX_STOPS && ImGui::SmallButton("add stop")) {
                    // Insert halfway between the last two stops, averaging their colours.
                    const int n = sp.stopCount;
                    sp.stopPos[n] = sp.stopPos[n - 1];
                    for (int c = 0; c < 3; ++c) sp.stopColor[n][c] = sp.stopColor[n - 1][c];
                    sp.stopPos[n - 1] = 0.5f * (sp.stopPos[n - 2] + sp.stopPos[n]);
                    for (int c = 0; c < 3; ++c)
                        sp.stopColor[n - 1][c] = 0.5f * (sp.stopColor[n - 2][c] + sp.stopColor[n][c]);
                    ++sp.stopCount;
                }
            } else {
                ImGui::TextDisabled("a + b cos(2pi(c t + d))");
                ImGui::DragFloat3("a", sp.a, 0.01f, -1.f, 2.f);
                ImGui::DragFloat3("b", sp.b, 0.01f, -1.f, 2.f);
                ImGui::DragFloat3("c", sp.c, 0.01f, -4.f, 4.f);
                ImGui::DragFloat3("d", sp.d, 0.01f, -1.f, 1.f);
            }
            ImGui::ColorEdit3("inside", sp.inside, ImGuiColorEditFlags_Float);

            ImGui::SeparatorText("effects");
            bool slopes = sp.slopes != 0;
            if (ImGui::Checkbox("slope light", &slopes)) sp.slopes = slopes;
            if (slopes) {
                ImGui::SliderFloat("angle", &sp.slopeAngle, 0.f, 360.f, "%.0f deg");
                ImGui::SliderFloat("height", &sp.slopeHeight, 0.2f, 4.f);
                ImGui::SliderFloat("strength", &sp.slopeStrength, 0.f, 1.f);
            }
            bool lines = sp.lines != 0;
            if (ImGui::Checkbox("iteration lines", &lines)) sp.lines = lines;
            if (lines) {
                ImGui::SliderFloat("per iter", &sp.lineDensity, 0.05f, 4.f, "%.2f",
                                   ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("width px", &sp.lineWidth, 0.5f, 4.f);
                ImGui::SliderFloat("opacity", &sp.lineStrength, 0.f, 1.f);
                ImGui::ColorEdit3("line colour", sp.lineColor, ImGuiColorEditFlags_Float);
            }
            ImGui::SliderFloat("edge (DE)", &sp.deStrength, 0.f, 4.f);
            ImGui::SliderFloat("exposure", &sp.exposure, 0.f, 4.f);

            ImGui::SeparatorText("animation");
            {
                bool an = app.animate;
                if (ImGui::Checkbox("animate", &an)) setAnimate(app, an);
            }
            ImGui::SliderFloat("palette cyc/s", &sp.cycleSpeed, -1.f, 1.f, "%.3f");
            ImGui::SliderFloat("light deg/s", &sp.lightSpeed, -90.f, 90.f, "%.1f");
            ImGui::SliderFloat("wave cyc/s", &sp.waveSpeed, -2.f, 2.f, "%.2f");
            ImGui::SliderFloat("settled fps cap", &app.bgFps, 0.f, 240.f, "%.0f (0 = every frame)");
        }
        drawVideoUi(app);
        if (ImGui::CollapsingHeader("Post")) {
            PostParams& pp = app.post;
            const char* tms[] = {"clamp", "Reinhard", "ACES"};
            ImGui::Combo("tone map", &pp.tonemap, tms, 3);
            ImGui::SliderFloat("HDR boost", &pp.hdrBoost, 0.25f, 8.f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SeparatorText("bloom");
            ImGui::SliderFloat("intensity", &pp.bloomIntensity, 0.f, 3.f);
            ImGui::SliderFloat("threshold", &pp.bloomThreshold, 0.f, 3.f);
            ImGui::SliderFloat("knee", &pp.bloomKnee, 0.f, 1.f);
            ImGui::SliderInt("radius", &pp.bloomRadius, 1, 24);
            ImGui::SliderFloat("wide level", &pp.bloomWide, 0.f, 2.f);
            ImGui::SeparatorText("lens");
            ImGui::SliderFloat("vignette", &pp.vignette, 0.f, 1.f);
            ImGui::SliderFloat("vignette soft", &pp.vignetteSoft, 0.05f, 1.2f);
            ImGui::SliderFloat("aberration px", &pp.aberration, 0.f, 12.f);
            ImGui::SliderFloat("grain", &pp.grain, 0.f, 1.f);
            ImGui::SliderFloat("sharpen", &pp.sharpen, 0.f, 2.f);
            ImGui::SeparatorText("grade");
            ImGui::SliderFloat("saturation", &pp.saturation, 0.f, 2.f);
            ImGui::SliderFloat("contrast", &pp.contrast, 0.5f, 2.f);
            ImGui::SliderFloat("gain", &pp.gain, 0.f, 4.f);
            if (ImGui::Button("reset post")) pp = PostParams();
        }
    }
    ImGui::End();
}

}  // namespace

int main(int argc, char** argv) {
    SDL_SetMainReady();
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // scripted runs read stdout after a crash too
    // Optional start location: mandelgpu <re> <im> <scale> [maxIter]
    // Numbers are decimal strings, any length. scale is complex units per pixel.
    // Optional: --script "wheel:5;wait:500;shot:out.png;quit" (see script.h).
    std::string argRe, argIm, scriptText;
    double argScale = 0.0;
    int argIter = 0;
    int argPreset = 0;
    int argSs = 1;
    bool argNoBla = false;
    bool argDouble = false;
    std::string argPost;
    std::string argShade;  // "offset=0.329,density=107.8,animate=0,cycle=-0.42" 
    std::string argLoad;  // saved preset name
    std::string argLoadFile;  // preset file path (screenshot sidecar)
    std::string argSave;  // save the starting parameters under this name
    std::string argAa;    // "pattern,filter,radius" 
    bool argFullscreen = false;
    bool argHidden = false;  // scripted runs: never show or focus the window  // "bloom=1.2,vignette=0.4,tonemap=2,..." 
    bool argVideo = false;
    VideoParams argVideoParams;
    int argWinW = 0, argWinH = 0;  // initial window size in pixels
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--script") == 0 && i + 1 < argc) {
            scriptText = argv[++i];
        } else if (std::strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            argPreset = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ss") == 0 && i + 1 < argc) {
            argSs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nobla") == 0) {
            argNoBla = true;
        } else if (std::strcmp(argv[i], "--double") == 0) {
            argDouble = true;
        } else if (std::strcmp(argv[i], "--post") == 0 && i + 1 < argc) {
            argPost = argv[++i];
        } else if (std::strcmp(argv[i], "--shade") == 0 && i + 1 < argc) {
            argShade = argv[++i];
        } else if (std::strcmp(argv[i], "--fullscreen") == 0) {
            argFullscreen = true;
        } else if (std::strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            argLoad = argv[++i];
        } else if (std::strcmp(argv[i], "--loadfile") == 0 && i + 1 < argc) {
            argLoadFile = argv[++i];
        } else if (std::strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            argSave = argv[++i];
        } else if (std::strcmp(argv[i], "--aa") == 0 && i + 1 < argc) {
            argAa = argv[++i];
        } else if (std::strcmp(argv[i], "--hidden") == 0) {
            argHidden = true;
        } else if (std::strcmp(argv[i], "--video") == 0) {
            argVideo = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') argVideoParams.outPath = argv[++i];
        } else if (std::strcmp(argv[i], "--video-seconds") == 0 && i + 1 < argc) {
            argVideoParams.seconds = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--video-fps") == 0 && i + 1 < argc) {
            argVideoParams.fps = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--video-size") == 0 && i + 1 < argc) {
            std::sscanf(argv[++i], "%dx%d", &argVideoParams.width, &argVideoParams.height);
        } else if (std::strcmp(argv[i], "--video-from") == 0 && i + 1 < argc) {
            argVideoParams.zoomFrom = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--video-sdr") == 0) {
            argVideoParams.hdr = false;
        } else if (std::strcmp(argv[i], "--video-cq") == 0 && i + 1 < argc) {
            argVideoParams.cq = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--video-hold") == 0 && i + 1 < argc) {
            std::sscanf(argv[++i], "%lf,%lf", &argVideoParams.holdStart, &argVideoParams.holdEnd);
        } else if (std::strcmp(argv[i], "--video-ease") == 0 && i + 1 < argc) {
            argVideoParams.ease = (float)std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--video-nits") == 0 && i + 1 < argc) {
            argVideoParams.nits = (float)std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--video-headroom") == 0 && i + 1 < argc) {
            argVideoParams.headroom = (float)std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--video-sub") == 0 && i + 1 < argc) {
            argVideoParams.subsamples = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--video-codec") == 0 && i + 1 < argc) {
            argVideoParams.codec = argv[++i];
        } else if (std::strcmp(argv[i], "--video-noramp") == 0) {
            argVideoParams.iterRamp = false;
        } else if (std::strcmp(argv[i], "--video-debug") == 0) {
            argVideoParams.debug = true;
        } else if (std::strcmp(argv[i], "--video-dumpkeys") == 0) {
            argVideoParams.dumpKeys = true;
        } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            std::sscanf(argv[++i], "%dx%d", &argWinW, &argWinH);
        } else {
            positional.push_back(argv[i]);
        }
    }
    if (positional.size() >= 3) {
        argRe = positional[0];
        argIm = positional[1];
        argScale = std::atof(positional[2].c_str());
        if (positional.size() >= 4) argIter = std::atoi(positional[3].c_str());
    }
    Script script;
    const bool scripted = !scriptText.empty() && script.parse(scriptText);
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    App app;
    float initialScale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    if (initialScale <= 0.f) initialScale = 1.f;
    app.window = SDL_CreateWindow("mandelbloom",
                                  argWinW > 0 ? argWinW : (int)(1280 * initialScale),
                                  argWinH > 0 ? argWinH : (int)(800 * initialScale),
                                  SDL_WINDOW_RESIZABLE |
                                      (argHidden ? (SDL_WINDOW_HIDDEN | SDL_WINDOW_NOT_FOCUSABLE)
                                                 : 0));
    if (!app.window) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }
    HWND hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(app.window),
                                             SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    {
        int pw0 = 0, ph0 = 0;
        SDL_GetWindowSizeInPixels(app.window, &pw0, &ph0);
        if (!app.display.init(hwnd, pw0, ph0)) return 1;
    }
    std::printf("D3D12 FP16 scRGB swapchain\n");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL3_InitForD3D(app.window);
    if (!app.display.imguiInit()) return 1;

    if (!app.renderer.init()) return 1;
    std::printf("CUDA: %s\n", app.renderer.deviceName());
    readHdrState(app);
    std::printf("HDR: %s, SDR white %.2f, headroom %.2f\n", app.hdrEnabled ? "on" : "off",
                app.sdrWhite, app.hdrHeadroom);
    applyPreset(app, argPreset);  // classic (0) sets the linear anchored transfer
    app.videoParams = argVideoParams;
    if (argVideo) {
        app.autoVideo = true;
        app.video.quitAfter = true;
    }
    if (!argLoad.empty()) {
        Preset pr;
        if (loadPreset(argLoad, pr)) {
            applyPresetTo(app, pr);
        } else {
            app.toast = "preset not found: " + argLoad;
            app.toastUntil = nowSeconds() + 8.0;
        }
    }
    if (!argLoadFile.empty()) {
        // Relative paths: the working directory, then beside the executable,
        // then the repo root above build/<preset>/ (so docs\cover.preset works
        // from anywhere).
        std::vector<std::string> tries = {argLoadFile};
        if (!std::filesystem::path(argLoadFile).is_absolute()) {
            const char* base = SDL_GetBasePath();
            if (base) {
                tries.push_back(std::string(base) + argLoadFile);
                tries.push_back(std::string(base) + "..\\..\\" + argLoadFile);
            }
        }
        Preset pr;
        bool loaded = false;
        for (const auto& t : tries) {
            std::error_code ec;
            if (std::filesystem::exists(t, ec) && loadPresetFile(t, pr)) {
                applyPresetTo(app, pr);
                loaded = true;
                break;
            }
        }
        if (!loaded) {
            app.toast = "preset file not found: " + argLoadFile;
            app.toastUntil = nowSeconds() + 8.0;
        }
    }
    if (!argAa.empty()) {
        int pat = 0, fil = 0;
        float rad = 0.75f;
        std::sscanf(argAa.c_str(), "%d,%d,%f", &pat, &fil, &rad);
        app.aa.pattern = std::min(std::max(pat, 0), AA_PATTERN_COUNT - 1);
        app.aa.filter = std::min(std::max(fil, 0), FILTER_COUNT - 1);
        app.aa.radius = rad;
        app.view.aaPattern = app.aa.pattern;
    }
    if (!argSave.empty()) {
        Preset cur;
        cur.shade = app.shade;
        cur.post = app.post;
        cur.animate = app.animate;
        std::printf("save preset %s: %s\n", argSave.c_str(), savePreset(argSave, cur) ? "ok" : "failed");
    }
    if (argFullscreen) setFullscreen(app, true);
    app.ss = std::min(3, std::max(1, argSs));
    if (argNoBla) app.view.useBla = false;
    if (argDouble) app.view.useFloat = false;
    if (!argShade.empty()) {
        size_t start = 0;
        while (start < argShade.size()) {
            size_t end = argShade.find(',', start);
            if (end == std::string::npos) end = argShade.size();
            const std::string kv = argShade.substr(start, end - start);
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) {
                const std::string k = kv.substr(0, eq);
                const float v = (float)std::atof(kv.substr(eq + 1).c_str());
                ShadeParams& sp = app.shade;
                if (k == "offset") sp.offset = v;
                else if (k == "density") sp.density = v;
                else if (k == "transfer") sp.transfer = (int)v;
                else if (k == "anchor") sp.anchor = (int)v;
                else if (k == "animate") app.animate = v != 0.f;
                else if (k == "cycle") sp.cycleSpeed = v;
                else if (k == "light") sp.lightSpeed = v;
                else if (k == "wave") sp.waveSpeed = v;
                else if (k == "exposure") sp.exposure = v;
                else if (k == "edge") sp.deStrength = v;
                else if (k == "mode") sp.mode = (int)v;
                else if (k == "special") sp.special = v;
            }
            start = end + 1;
        }
    }
    if (!argPost.empty()) {
        // Comma-separated key=value pairs.
        size_t start = 0;
        while (start < argPost.size()) {
            size_t end = argPost.find(',', start);
            if (end == std::string::npos) end = argPost.size();
            const std::string kv = argPost.substr(start, end - start);
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) {
                const std::string k = kv.substr(0, eq);
                const float v = (float)std::atof(kv.substr(eq + 1).c_str());
                PostParams& pp = app.post;
                if (k == "tonemap") pp.tonemap = (int)v;
                else if (k == "boost") pp.hdrBoost = v;
                else if (k == "bloom") pp.bloomIntensity = v;
                else if (k == "threshold") pp.bloomThreshold = v;
                else if (k == "radius") pp.bloomRadius = (int)v;
                else if (k == "wide") pp.bloomWide = v;
                else if (k == "vignette") pp.vignette = v;
                else if (k == "aberration") pp.aberration = v;
                else if (k == "grain") pp.grain = v;
                else if (k == "saturation") pp.saturation = v;
                else if (k == "contrast") pp.contrast = v;
                else if (k == "gain") pp.gain = v;
                else if (k == "sharpen") pp.sharpen = v;
            }
            start = end + 1;
        }
    }

    Uint64 lastTick = SDL_GetPerformanceCounter();
    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (!handleEvent(app, e)) running = false;
        }

        const Uint64 now = SDL_GetPerformanceCounter();
        const double dt = (double)(now - lastTick) / (double)SDL_GetPerformanceFrequency();
        lastTick = now;
        if (dt > 0) app.fps = 0.9 * app.fps + 0.1 * (1.0 / dt);
        if (app.animate) app.animTime += (float)dt;
        applyInertia(app, dt);

        int pw = 0, ph = 0;
        SDL_GetWindowSizeInPixels(app.window, &pw, &ph);
        // The previous frame's copy out of the shared buffer must finish
        // before anything writes into it again.
        app.display.waitForGpu();
        const bool resized = app.display.resize(pw, ph);
        if (resized && app.video.active) app.video.previewOk = false;  // display buffer replaced
        if (!app.video.active && (resized || app.ssApplied != app.ss)) {
            const bool first = app.view.width == 0;
            app.view.width = pw * app.ss;
            app.view.height = ph * app.ss;
            app.ssApplied = app.ss;
            syncViewScale(app);
            if (first) {
                resetView(app);
                if (argScale > 0.0) {
                    app.render.scale = argScale;
                    syncViewScale(app);
                    updatePrecision(app);
                    app.render.cx.set(argRe);
                    app.render.cy.set(argIm);
                    if (argIter > 0) app.view.maxIter = argIter;
                    snapShown(app);
                }
            }
            app.renderer.bindOutput(app.display.sharedHandle(), app.display.sharedSize(),
                                    app.display.rowPitch(), pw, ph, app.ss);
            app.gens.clear();
            app.dirty = true;
        }
        if (pw <= 0 || ph <= 0) continue;
        if (app.autoVideo && app.view.width > 0 && !app.video.active) {
            app.autoVideo = false;
            if (!startVideo(app)) running = false;
        }

        if (app.video.active) {
            videoStep(app);
            if (!app.video.active && app.video.quitAfter) running = false;
        } else {
        // View changes are applied only between slices, so the slice in
        // flight finishes (its samples stay valid for its generation) and the
        // main thread never blocks on the GPU.
        const bool gpuIdle = !app.renderer.iterateBusy();
        if (gpuIdle && app.panDirty && !app.dirty) {
            // Shift in whole 8-pixel steps so the coarse-level anchors stay on
            // their grid; the remainder stays as an offset between the target
            // view and the field placement until the next shift.
            const int qx = (app.panDx / 8) * 8, qy = (app.panDy / 8) * 8;
            if (qx != 0 || qy != 0) {
                // The field moves under every generation: their centres move
                // by the shifted amount at their own scale. Do this first: the
                // renderer snapshots app.view (with the reference offset) when
                // it resumes, so the offset must already describe the shifted
                // field.
                for (auto& g : app.gens) {
                    g.view.cx.addDouble(qx * (g.view.scale / app.ss));
                    g.view.cy.subDouble(qy * (g.view.scale / app.ss));
                }
                app.panDx -= qx;
                app.panDy -= qy;
                updateRefOffset(app);
                app.renderer.shiftAndResume(app.view, qx, qy);
                if (std::hypot(app.view.refOffX, app.view.refOffY) > halfDiagonal(app)) {
                    rebuildReference(app);
                    app.renderer.restartPending(app.view);
                } else {
                    rebuildBla(app);
                }
            }
            app.panDirty = false;
        }
        if (gpuIdle && app.dirty) {
            // New generation: remember its view; older ones keep theirs. Any
            // pending pan is folded into the new view (the field did not move).
            App::Gen g;
            g.view = app.render;
            g.id = app.nextGen;
            app.nextGen = app.nextGen % 65535 + 1;  // 16-bit ids, never 0
            app.gens.insert(app.gens.begin(), g);
            if (app.gens.size() > MAX_GENS) app.gens.resize(MAX_GENS);
            rebuildReference(app);
            app.renderer.beginIterate(app.view, g.id);
            app.panDx = app.panDy = 0;
            app.panDirty = false;
            app.dirty = false;
        }
        tweenShown(app, dt);

        // Keep one slice in flight. When a level completes under a reference
        // that escaped early, re-reference at a pixel that outlived it.
        if (!app.renderer.iterateBusy()) {
            const bool sliceDone = app.renderer.takeSliceFinished();
            const int done = app.renderer.completedStride();
            if (sliceDone && app.ref.escaped && done > 0 && done != app.rerefCheckedStride &&
                app.rerefRounds < 12) {
                app.rerefCheckedStride = done;
                int fx, fy;
                if (app.renderer.findUnreliable(fx, fy)) rereferenceAt(app, fx, fy);
            }
            if (!app.renderer.iterateDone()) app.renderer.stepIterate();
        }

        // Display. While anything moves, composite every frame from the
        // field. Once settled, resolve the subsamples once into a cache and
        // recolour only when a parameter or the animation time changes.
        {
            CompositeMap m;
            m.ss = app.ss;
            m.genCount = 0;
            for (const auto& g : app.gens) {
                if (m.genCount >= MAX_GENS) break;
                GenMap& gm = m.gens[m.genCount++];
                mapOnto(app, g.view, app.ss, app.view.width, app.view.height, gm.ox, gm.oy,
                        gm.ratio);
                gm.pixelScale = (float)(g.view.scale / app.ss);
                gm.gen = g.id;
            }
            // Palette anchor: the smallest escape iteration of the newest
            // generation that has one.
            {
                const float cur = app.renderer.minIter();
                if (cur >= 0.f)
                    for (auto& g : app.gens)
                        if (g.id == app.renderer.currentGen()) g.minIter = cur;
                float base = 0.f;
                for (const auto& g : app.gens)
                    if (g.minIter >= 0.f) { base = std::floor(g.minIter); break; }
                app.shade.iterBase = app.shade.anchor ? base : 0.f;
            }
            const bool paletteChanged = !app.lastShadeValid || !paletteEqual(app.shade, app.lastShade);
            const bool staticChanged = !app.lastShadeValid || !staticEqual(app.shade, app.lastShade);
            const bool dynamicChanged = !app.lastShadeValid || !dynamicEqual(app.shade, app.lastShade);
            if (paletteChanged) app.renderer.buildPaletteLut(app.shade);

            const bool settled = app.renderer.iterateDone() && !app.tweening && !app.inertia &&
                                 !app.panDirty && !app.dirty && !app.gens.empty();
            const int gen0 = app.gens.empty() ? -1 : app.gens[0].id;
            if (!settled) {
                app.cacheValid = false;
                app.cachedShown = false;
                app.renderer.composite(app.shade, app.animTime, m);
                app.imageDirty = true;
            } else {
                if (!app.cacheValid || staticChanged || app.cacheGen != gen0) {
                    app.renderer.buildShadeCache(app.shade, m);
                    app.cacheValid = true;
                    app.cacheGen = gen0;
                    app.cachedShown = false;
                }
                bool need = !app.cachedShown || dynamicChanged || paletteChanged;
                if (app.animate) {
                    const double now = nowSeconds();
                    if (app.bgFps <= 0.f || now - app.lastShadeSec >= 1.0 / app.bgFps) need = true;
                }
                if (need) {
                    if (app.renderer.shadeCached(app.shade, app.animTime, app.aa)) {
                        app.cachedShown = true;
                        app.lastShadeSec = nowSeconds();
                        app.imageDirty = true;
                    }
                }
            }
            app.lastShade = app.shade;
            app.lastShadeValid = true;

            // Post-processing: whenever the image or its parameters changed,
            // and every frame while grain animates.
            const bool postChanged = !app.lastPostValid ||
                                     std::memcmp(&app.post, &app.lastPost, sizeof(PostParams)) != 0;
            if (app.imageDirty || postChanged || (app.post.grain > 0.f && app.animate)) {
                app.renderer.postProcess(app.post, app.frameIndex);
                app.imageDirty = false;
            }
            app.lastPost = app.post;
            app.lastPostValid = true;
            ++app.frameIndex;
        }
        }  // interactive path

        app.display.imguiNewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        drawUi(app);
        if (!app.toast.empty() && nowSeconds() < app.toastUntil) {
            ImGui::SetNextWindowPos(ImVec2(pw * 0.5f, ph - 40 * app.uiScale), ImGuiCond_Always,
                                    ImVec2(0.5f, 1.f));
            ImGui::SetNextWindowBgAlpha(0.8f);
            ImGui::Begin("##toast", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoSavedSettings);
            ImGui::TextUnformatted(app.toast.c_str());
            ImGui::End();
        }
        ImGui::Render();

        // CUDA must be done writing the output before D3D12 copies it.
        app.renderer.syncDisplay();

        if (scripted) {
            bool quit = false;
            {
                int pdx, pdy;
                if (script.takePan(pdx, pdy)) panPixels(app, pdx, pdy);
                const int an = script.takeAnimate();
                if (an >= 0) setAnimate(app, an != 0);
                const int sr = script.takeScreenshot();
                if (sr) app.screenshotRequest = sr;
            }
            const std::string shot = script.tick(
                (double)SDL_GetPerformanceCounter() / (double)SDL_GetPerformanceFrequency(), pw,
                ph, quit);
            if (!shot.empty()) {
                // The composited image straight from the pixel buffer, plus
                // the numbers the overlay would show, on stdout.
                std::vector<uint32_t> px;
                if (!app.renderer.readOutput(px)) {
                    std::printf("shot: readOutput failed\n");
                } else if (!writePng(shot.c_str(), px.data(), pw, ph)) {
                    std::printf("shot: writePng failed for %s\n", shot.c_str());
                }
                const double zoom = 3.2 / (app.render.scale * (app.view.width / app.ss));
                const int dg = std::max(5, (int)std::ceil(-std::log10(app.render.scale)) + 3);
                std::printf("shot %s zoom=%.3g iterate=%.0fms done=%d stride=%d/%d tween=%d "
                            "inertia=%d display=%.3fms cached=%d fps=%.0f progress=%.2f eta=%.0f centre=%s %s\n",
                            shot.c_str(), zoom, app.renderer.lastIterateMs(),
                            app.renderer.iterateDone() ? 1 : 0, app.renderer.currentStride(),
                            app.renderer.completedStride(), app.tweening ? 1 : 0,
                            app.inertia ? 1 : 0, app.renderer.lastShadeMs(),
                            app.cachedShown ? 1 : 0, app.fps, app.renderer.progress(),
                            app.renderer.etaMs(), app.render.cx.toString(dg).c_str(),
                            app.render.cy.toString(dg).c_str());
                std::printf("  ref iters=%d escaped=%d rerefs=%d base=%.0f transfer=%d anchor=%d\n", app.ref.length,
                            app.ref.escaped ? 1 : 0, app.rerefRounds,
                            app.shade.iterBase, app.shade.transfer, app.shade.anchor);
                std::printf("  shown scale=%.6g render scale=%.6g gens=%zu\n", app.shown.scale,
                            app.render.scale, app.gens.size());
                if (!app.gens.empty()) {
                    CudaRenderer::DebugStats ds;
                    app.renderer.debugStats(app.gens[0].id, ds);
                    std::printf("  field: total=%d genMatch=%d (inside %d) genZero=%d genOther=%d | "
                                "active=%d escaped=%d insideSt=%d\n",
                                ds.total, ds.genMatch, ds.genMatchInside, ds.genZero, ds.genOther,
                                ds.stActive, ds.stEscaped, ds.stInside);
                }
                for (const auto& g : app.gens) {
                    double ox, oy, ratio;
                    mapOnto(app, g.view, app.ss, app.view.width, app.view.height, ox, oy, ratio);
                    std::printf("  gen %d ox=%.1f oy=%.1f ratio=%.4f\n", g.id, ox, oy, ratio);
                }
                std::fflush(stdout);
            }
            if (quit) {
                if (app.video.active) videoFinish(app, true);  // keeps the frames so far
                running = false;
            }
        }
        if (app.screenshotRequest == 3) app.display.requestCapture();
        app.display.present(ImGui::GetDrawData(), app.vsync, app.sdrWhite);
        if (app.screenshotRequest == 3) takeUiScreenshot(app);
        else if (app.screenshotRequest) takeScreenshot(app, app.screenshotRequest == 2);
        app.screenshotRequest = 0;
    }

    app.display.imguiShutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    app.display.shutdown();
    SDL_DestroyWindow(app.window);
    SDL_Quit();
    return 0;
}
