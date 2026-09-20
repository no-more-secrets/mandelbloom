#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_opengl3.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "bigfloat.h"
#include "bla.h"
#include "gl_display.h"
#include "image_io.h"
#include "script.h"
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
    SDL_GLContext gl = nullptr;
    GlDisplay display;
    CudaRenderer renderer;

    // Three views. render: where the pass computes (the target). shown: what
    // the display shows, tweened toward render. last: the last finished
    // frame kept on the GPU for reprojection.
    ViewParams view;   // render pass grid + scale
    View render;
    View shown;
    View last;
    bool lastValid = false;
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
    float blaEpsLog2 = -24.f;
    ShadeParams shade;
    bool dirty = true;  // render view changed: restart the pass
    bool animate = false;
    float animSpeed = 0.05f;  // palette cycles per second
    float animTime = 0.f;
    bool dragging = false;
    // Pan inertia: velocity in pixels per second, carried after release.
    double velX = 0, velY = 0;
    double inertiaAccX = 0, inertiaAccY = 0;  // sub-pixel remainder
    double lastMotionSec = 0;
    bool inertia = false;
    bool inertiaEnabled = true;
    float inertiaTau = 0.25f;  // seconds for the velocity to fall to 37%
    bool showUi = true;
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
    const double logRatio = std::log(app.render.scale / app.shown.scale);
    app.shown.scale *= std::exp(logRatio * k);
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
void updateRefOffset(App& app) {
    app.view.refOffX = BigFloat::diff(app.render.cx, app.refCx);
    app.view.refOffY = BigFloat::diff(app.render.cy, app.refCy);
}

void rebuildBla(App& app) {
    const double cMax = halfDiagonal(app) + std::hypot(app.view.refOffX, app.view.refOffY);
    buildBla(app.ref, cMax, std::exp2((double)app.blaEpsLog2), app.bla);
    app.renderer.uploadBla(app.bla.nodes.data(), (int)app.bla.nodes.size(),
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
void applyPreset(ShadeParams& sp, int which) {
    sp = ShadeParams();
    switch (which) {
        case 1: {  // Pastel lines: pale palette, relief lighting, thin dark lines
            const float a[3] = {0.86f, 0.84f, 0.88f}, b[3] = {0.14f, 0.16f, 0.12f};
            const float d[3] = {0.00f, 0.25f, 0.55f};
            for (int i = 0; i < 3; ++i) { sp.a[i] = a[i]; sp.b[i] = b[i]; sp.d[i] = d[i]; }
            sp.density = 24.f;
            sp.slopes = 1;
            sp.slopeAngle = 60.f;
            sp.slopeHeight = 1.2f;
            sp.slopeStrength = 0.55f;
            sp.lines = 1;
            sp.lineDensity = 1.f;
            sp.lineWidth = 1.1f;
            sp.lineStrength = 0.8f;
            sp.deStrength = 0.6f;
            sp.inside[0] = sp.inside[1] = sp.inside[2] = 0.05f;
            break;
        }
        case 2: {  // Relief: classic colours with strong slope lighting
            sp.slopes = 1;
            sp.slopeAngle = 30.f;
            sp.slopeHeight = 1.0f;
            sp.slopeStrength = 0.85f;
            sp.deStrength = 0.f;
            break;
        }
        case 3: {  // Mono lines: white paper, ink lines, soft relief
            for (int i = 0; i < 3; ++i) { sp.a[i] = 0.93f; sp.b[i] = 0.05f; }
            sp.slopes = 1;
            sp.slopeStrength = 0.35f;
            sp.lines = 1;
            sp.lineWidth = 1.0f;
            sp.lineStrength = 0.9f;
            sp.deStrength = 0.4f;
            break;
        }
        default:
            break;
    }
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
        case SDL_EVENT_KEY_DOWN:
            if (io.WantCaptureKeyboard) break;
            if (e.key.key == SDLK_ESCAPE) return false;
            if (e.key.key == SDLK_R) resetView(app);
            if (e.key.key == SDLK_TAB) app.showUi = !app.showUi;
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
                const double dt = std::max(t - app.lastMotionSec, 0.002);
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
        ImGui::Text("display %.2f ms", app.renderer.lastShadeMs());
        ImGui::Text("frame   %.0f fps", app.fps);
        ImGui::Checkbox("smooth zoom", &app.tweenEnabled);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100 * app.uiScale);
        ImGui::SliderFloat("##tau", &app.tweenTau, 0.02f, 0.3f, "%.2f s");
        ImGui::Checkbox("pan inertia", &app.inertiaEnabled);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100 * app.uiScale);
        ImGui::SliderFloat("##itau", &app.inertiaTau, 0.05f, 1.0f, "%.2f s");
        ImGui::TextDisabled("drag: pan  wheel: zoom  R: reset  Tab: hide");

        if (ImGui::CollapsingHeader("Shading")) {
            ShadeParams& sp = app.shade;
            ImGui::TextDisabled("presets");
            ImGui::SameLine();
            if (ImGui::SmallButton("classic")) applyPreset(sp, 0);
            ImGui::SameLine();
            if (ImGui::SmallButton("pastel lines")) applyPreset(sp, 1);
            ImGui::SameLine();
            if (ImGui::SmallButton("relief")) applyPreset(sp, 2);
            ImGui::SameLine();
            if (ImGui::SmallButton("mono lines")) applyPreset(sp, 3);
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
            ImGui::Checkbox("animate", &app.animate);
            ImGui::SameLine();
            ImGui::SliderFloat("speed", &app.animSpeed, -1.f, 1.f, "%.3f cyc/s");
            ImGui::SliderFloat("density", &sp.density, 4.f, 1024.f, "%.1f",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("offset", &sp.offset, 0.f, 1.f);
            bool logScale = sp.logScale != 0;
            if (ImGui::Checkbox("log scale", &logScale)) sp.logScale = logScale;
            ImGui::SliderFloat("edge (DE)", &sp.deStrength, 0.f, 4.f);
            ImGui::SliderFloat("exposure", &sp.exposure, 0.f, 4.f);
            ImGui::ColorEdit3("inside", sp.inside, ImGuiColorEditFlags_Float);
            ImGui::TextDisabled("palette a + b cos(2pi(c t + d))");
            ImGui::DragFloat3("a", sp.a, 0.01f, -1.f, 2.f);
            ImGui::DragFloat3("b", sp.b, 0.01f, -1.f, 2.f);
            ImGui::DragFloat3("c", sp.c, 0.01f, -4.f, 4.f);
            ImGui::DragFloat3("d", sp.d, 0.01f, -1.f, 1.f);
            if (ImGui::Button("reset palette")) sp = ShadeParams();
        }
    }
    ImGui::End();
}

}  // namespace

int main(int argc, char** argv) {
    SDL_SetMainReady();
    // Optional start location: mandelgpu <re> <im> <scale> [maxIter]
    // Numbers are decimal strings, any length. scale is complex units per pixel.
    // Optional: --script "wheel:5;wait:500;shot:out.png;quit" (see script.h).
    std::string argRe, argIm, scriptText;
    double argScale = 0.0;
    int argIter = 0;
    int argPreset = 0;
    int argSs = 1;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--script") == 0 && i + 1 < argc) {
            scriptText = argv[++i];
        } else if (std::strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            argPreset = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ss") == 0 && i + 1 < argc) {
            argSs = std::atoi(argv[++i]);
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
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    App app;
    float initialScale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    if (initialScale <= 0.f) initialScale = 1.f;
    app.window = SDL_CreateWindow("mandelgpu", (int)(1280 * initialScale),
                                  (int)(800 * initialScale),
                                  SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!app.window) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }
    app.gl = SDL_GL_CreateContext(app.window);
    if (!app.gl) {
        std::fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_MakeCurrent(app.window, app.gl);
    SDL_GL_SetSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
        std::fprintf(stderr, "gladLoadGLLoader failed\n");
        return 1;
    }
    std::printf("OpenGL %s on %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL3_InitForOpenGL(app.window, app.gl);
    ImGui_ImplOpenGL3_Init("#version 460");
    applyUiScale(app);

    if (!app.renderer.init()) return 1;
    std::printf("CUDA: %s\n", app.renderer.deviceName());
    if (argPreset > 0) applyPreset(app.shade, argPreset);
    app.ss = std::min(3, std::max(1, argSs));

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
        if (app.animate) app.animTime += (float)dt * app.animSpeed;
        applyInertia(app, dt);

        int pw = 0, ph = 0;
        SDL_GetWindowSizeInPixels(app.window, &pw, &ph);
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
            app.renderer.bindPixelBuffer(app.display.pbo(), pw, ph, app.ss);
            app.lastValid = false;
            app.dirty = true;
        }
        if (pw <= 0 || ph <= 0) continue;

        if (app.panDirty && !app.dirty) {
            // Shift what we have, keep the reference unless we drifted more
            // than a view radius from it, and continue on the exposed strip.
            updateRefOffset(app);
            app.renderer.shiftAndResume(app.view, app.panDx, app.panDy);
            if (std::hypot(app.view.refOffX, app.view.refOffY) > halfDiagonal(app)) {
                rebuildReference(app);
                app.renderer.restartPending(app.view);
            } else {
                rebuildBla(app);
            }
            app.panDx = app.panDy = 0;
            app.panDirty = false;
        }
        if (app.dirty) {
            rebuildReference(app);
            app.renderer.beginIterate(app.view);
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

        // Display: composite the running field and the last finished frame
        // onto the shown view. Cheap, runs every frame on its own stream.
        {
            CompositeMap m;
            m.ss = app.ss;
            mapOnto(app, app.render, app.ss, app.view.width, app.view.height, m.nox, m.noy,
                    m.ratioN);
            const int done = app.renderer.completedStride();
            m.newDetail = app.renderer.iterateDone() ? (float)m.ratioN
                          : done > 0                 ? (float)(m.ratioN / done)
                                                     : 0.f;
            if (app.lastValid) {
                mapOnto(app, app.last, 1, pw, ph, m.oox, m.ooy, m.ratioO);
                m.oldDetail = (float)m.ratioO;
            }
            m.pixelScaleN = (float)app.view.scale;
            const bool settled = app.renderer.iterateDone() && !app.tweening;
            m.snapshot = settled ? 1 : 0;
            if (app.renderer.composite(app.shade, app.animTime, m)) {
                app.display.upload();
                if (settled) {
                    app.last.cx = app.shown.cx;
                    app.last.cy = app.shown.cy;
                    app.last.scale = app.shown.scale;
                    app.lastValid = true;
                }
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        app.display.draw(0.f, 0.f, 1.f, 1.f);
        drawUi(app);
        ImGui::Render();

        glViewport(0, 0, pw, ph);
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (scripted) {
            bool quit = false;
            const std::string shot = script.tick(
                (double)SDL_GetPerformanceCounter() / (double)SDL_GetPerformanceFrequency(), pw,
                ph, quit);
            if (!shot.empty()) {
                // The composited image straight from the pixel buffer, plus
                // the numbers the overlay would show, on stdout.
                std::vector<uint32_t> px((size_t)pw * ph);
                if (app.display.readPixels(px.data(), (int)px.size())) {
                    writePng(shot.c_str(), px.data(), pw, ph);
                }
                const double zoom = 3.2 / (app.render.scale * (app.view.width / app.ss));
                const int dg = std::max(5, (int)std::ceil(-std::log10(app.render.scale)) + 3);
                std::printf("shot %s zoom=%.3g iterate=%.0fms done=%d stride=%d/%d tween=%d "
                            "inertia=%d centre=%s %s\n",
                            shot.c_str(), zoom, app.renderer.lastIterateMs(),
                            app.renderer.iterateDone() ? 1 : 0, app.renderer.currentStride(),
                            app.renderer.completedStride(), app.tweening ? 1 : 0,
                            app.inertia ? 1 : 0, app.render.cx.toString(dg).c_str(),
                            app.render.cy.toString(dg).c_str());
                std::fflush(stdout);
            }
            if (quit) running = false;
        }
        SDL_GL_SwapWindow(app.window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DestroyContext(app.gl);
    SDL_DestroyWindow(app.window);
    SDL_Quit();
    return 0;
}
