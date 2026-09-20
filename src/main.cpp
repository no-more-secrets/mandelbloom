#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

namespace {

// A view of the plane: centre at high precision, pixel scale.
struct View {
    BigFloat cx{64}, cy{64};
    double scale = 1.0;
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
    };
    std::vector<Gen> gens;
    int nextGen = 1;
    int ss = 1;          // supersampling factor per axis (field = display * ss)
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
    char presetName[64] = "my preset";
    std::vector<std::string> savedPresets;
    bool savedPresetsValid = false;
    double fps = 0.0;
    float uiScale = 1.f;
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
}

void rebuildBla(App& app) {
    const double cMax = halfDiagonal(app) + std::hypot(app.view.refOffX, app.view.refOffY);
    buildBla(app.ref, cMax, std::exp2((double)app.blaEpsLog2), app.bla);
    buildBlaFloat(app.bla, app.blaF);
    app.renderer.uploadBla(app.bla.nodes.data(), app.blaF.data(), (int)app.bla.nodes.size(),
                           app.bla.levelOffset.data(), app.bla.levels, app.bla.steps);
}

void rebuildReference(App& app) {
    app.refCx = app.render.cx;
    app.refCy = app.render.cy;
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
           a.logScale == b.logScale && a.special == b.special && a.deStrength == b.deStrength &&
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
            if (e.key.key == SDLK_ESCAPE) return false;
            if (e.key.key == SDLK_R) resetView(app);
            if (e.key.key == SDLK_TAB) app.showUi = !app.showUi;
            if (e.key.key == SDLK_F11) setFullscreen(app, !app.fullscreen);
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

void drawUi(App& app) {
    if (!app.showUi) return;
    ImGui::SetNextWindowPos(ImVec2(10 * app.uiScale, 10 * app.uiScale), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.75f);
    if (ImGui::Begin("mandelgpu", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
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
        }
        if (ImGui::SliderInt("max iter", &app.view.maxIter, 64, 65536, "%d",
                             ImGuiSliderFlags_Logarithmic)) {
            app.dirty = true;
        }
        ImGui::Separator();
        ImGui::Text("ref     %.2f ms  (%d iters%s)", app.ref.computeMs, app.ref.length,
                    app.ref.escaped ? ", escaped" : "");
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
            ImGui::Text("iterate %.0f ms  (stride %d, %d / %d, slice %.1f ms)",
                        app.renderer.lastIterateMs(), app.renderer.currentStride(),
                        std::min(app.renderer.iterateProgress(), app.view.maxIter),
                        app.view.maxIter, app.renderer.lastSliceMs());
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

        if (ImGui::CollapsingHeader("Shading")) {
            ShadeParams& sp = app.shade;
            ImGui::TextDisabled("presets");
            for (int i = 0; i < BUILTIN_PRESET_COUNT; ++i) {
                if (i % 4 != 0) ImGui::SameLine();
                if (ImGui::SmallButton(kBuiltinPresetNames[i])) applyPreset(app, i);
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
            ImGui::SliderFloat("density", &sp.density, 4.f, 1024.f, "%.1f",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("offset", &sp.offset, 0.f, 1.f);
            bool logScale = sp.logScale != 0;
            if (ImGui::Checkbox("log scale", &logScale)) sp.logScale = logScale;

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
            ImGui::Checkbox("animate", &app.animate);
            ImGui::SliderFloat("palette cyc/s", &sp.cycleSpeed, -1.f, 1.f, "%.3f");
            ImGui::SliderFloat("light deg/s", &sp.lightSpeed, -90.f, 90.f, "%.1f");
            ImGui::SliderFloat("wave cyc/s", &sp.waveSpeed, -2.f, 2.f, "%.2f");
            ImGui::SliderFloat("settled fps cap", &app.bgFps, 0.f, 240.f, "%.0f (0 = every frame)");
        }
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
    std::string argLoad;  // saved preset name
    std::string argSave;  // save the starting parameters under this name
    bool argFullscreen = false;  // "bloom=1.2,vignette=0.4,tonemap=2,..." 
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
        } else if (std::strcmp(argv[i], "--fullscreen") == 0) {
            argFullscreen = true;
        } else if (std::strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            argLoad = argv[++i];
        } else if (std::strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            argSave = argv[++i];
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
    app.window = SDL_CreateWindow("mandelgpu", (int)(1280 * initialScale),
                                  (int)(800 * initialScale),
                                  SDL_WINDOW_RESIZABLE);
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
    if (argPreset > 0) applyPreset(app, argPreset);
    if (!argLoad.empty()) {
        Preset pr;
        if (loadPreset(argLoad, pr)) applyPresetTo(app, pr);
        else std::fprintf(stderr, "preset not found: %s\n", argLoad.c_str());
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
        if (resized || app.ssApplied != app.ss) {
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

        // Keep one slice in flight.
        if (!app.renderer.iterateBusy()) {
            app.renderer.takeSliceFinished();
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
                    if (app.renderer.shadeCached(app.shade, app.animTime)) {
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

        app.display.imguiNewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        drawUi(app);
        ImGui::Render();

        // CUDA must be done writing the output before D3D12 copies it.
        app.renderer.syncDisplay();

        if (scripted) {
            bool quit = false;
            {
                int pdx, pdy;
                if (script.takePan(pdx, pdy)) panPixels(app, pdx, pdy);
                const int an = script.takeAnimate();
                if (an >= 0) app.animate = an != 0;
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
                            "inertia=%d display=%.3fms cached=%d fps=%.0f centre=%s %s\n",
                            shot.c_str(), zoom, app.renderer.lastIterateMs(),
                            app.renderer.iterateDone() ? 1 : 0, app.renderer.currentStride(),
                            app.renderer.completedStride(), app.tweening ? 1 : 0,
                            app.inertia ? 1 : 0, app.renderer.lastShadeMs(),
                            app.cachedShown ? 1 : 0, app.fps, app.render.cx.toString(dg).c_str(),
                            app.render.cy.toString(dg).c_str());
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
            if (quit) running = false;
        }
        app.display.present(ImGui::GetDrawData(), app.vsync, app.sdrWhite);
    }

    app.display.imguiShutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    app.display.shutdown();
    SDL_DestroyWindow(app.window);
    SDL_Quit();
    return 0;
}
