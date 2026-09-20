#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_opengl3.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include "bigfloat.h"
#include "bla.h"
#include "gl_display.h"
#include "reference.h"
#include "render_cuda.h"

namespace {

struct App {
    SDL_Window* window = nullptr;
    SDL_GLContext gl = nullptr;
    GlDisplay display;
    CudaRenderer renderer;
    ViewParams view;
    BigFloat cx{64}, cy{64};   // view centre = reference parameter
    ReferenceOrbit ref;
    BlaTable bla;
    float blaEpsLog2 = -24.f;
    ShadeParams shade;
    bool dirty = true;        // view changed: re-iterate and re-shade
    bool shadeDirty = true;   // only coloring changed
    // What the display texture currently holds, so it can be reprojected
    // while a new pass runs.
    struct Shown {
        bool valid = false;
        BigFloat cx{64}, cy{64};
        double scale = 1.0;
        int width = 0, height = 0;
    } shown;
    bool showingNewPass = false;  // texture follows the running pass
    bool animate = false;
    float animSpeed = 0.05f;  // palette cycles per second
    float animTime = 0.f;
    bool dragging = false;
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
    const mpfr_prec_t p = precisionFor(app.view.scale);
    if (p != app.cx.prec()) {
        app.cx.setPrec(p);
        app.cy.setPrec(p);
    }
}

void zoomAt(App& app, float mx, float my, double factor) {
    // Keep the complex point under the cursor fixed while scaling.
    const double hx = 0.5 * app.view.width, hy = 0.5 * app.view.height;
    const double px = mx - hx, py = my - hy;
    const double oldScale = app.view.scale;
    app.view.scale *= factor;
    updatePrecision(app);
    app.cx.addDouble(px * (oldScale - app.view.scale));
    app.cy.subDouble(py * (oldScale - app.view.scale));
    app.dirty = true;
}

void resetView(App& app) {
    app.view.scale = 3.2 / std::max(1, app.view.width);
    updatePrecision(app);
    app.cx.set(-0.5);
    app.cy.set(0.0);
    app.dirty = true;
}

// Texture coordinates that place the shown frame under the current view.
void reprojectUv(const App& app, float& u0, float& v0, float& u1, float& v1) {
    const auto& sh = app.shown;
    if (!sh.valid || app.showingNewPass) {
        u0 = v0 = 0.f;
        u1 = v1 = 1.f;
        return;
    }
    const double ratio = app.view.scale / sh.scale;  // new pixel in old pixels
    const double dx = BigFloat::diff(app.cx, sh.cx) / sh.scale;   // centre shift, old px
    const double dy = -BigFloat::diff(app.cy, sh.cy) / sh.scale;
    const double ox = 0.5 * sh.width + dx, oy = 0.5 * sh.height + dy;
    const double hx = 0.5 * app.view.width, hy = 0.5 * app.view.height;
    u0 = (float)((ox - hx * ratio) / sh.width);
    v0 = (float)((oy - hy * ratio) / sh.height);
    u1 = (float)((ox + hx * ratio) / sh.width);
    v1 = (float)((oy + hy * ratio) / sh.height);
}

void markShown(App& app) {
    app.shown.valid = true;
    app.shown.cx = app.cx;
    app.shown.cy = app.cy;
    app.shown.scale = app.view.scale;
    app.shown.width = app.view.width;
    app.shown.height = app.view.height;
}

// Switch the display from the reprojected old frame to the running pass once
// the pass has at least as much detail as the stretched old frame.
bool newPassWorthShowing(const App& app) {
    const int done = app.renderer.completedStride();
    if (done == 0) return false;
    if (!app.shown.valid) return true;
    const double magnification = app.shown.scale / app.view.scale;  // >1 zoomed in
    // Zoomed in: wait until the new pass beats the stretched old frame.
    // Panned or zoomed out: the old frame leaves black edges, so accept the
    // half-resolution level rather than waiting for the full one.
    const double threshold = magnification > 1.0 ? magnification : 2.0;
    return (double)done <= threshold;
}

// Recompute the reference orbit at the current centre, build its BLA
// table for this view, and hand both to the GPU.
void rebuildReference(App& app) {
    computeReference(app.cx, app.cy, app.view.maxIter, 65536.0, app.ref);
    app.renderer.uploadReference(app.ref.zr.data(), app.ref.zi.data(), app.ref.length,
                                 app.ref.escaped);
    const double cMax = 0.5 * std::hypot((double)app.view.width, (double)app.view.height) *
                        app.view.scale;
    buildBla(app.ref, cMax, std::exp2((double)app.blaEpsLog2), app.bla);
    app.renderer.uploadBla(app.bla.nodes.data(), (int)app.bla.nodes.size(),
                           app.bla.levelOffset.data(), app.bla.levels, app.bla.steps);
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
            if (e.button.button == SDL_BUTTON_LEFT) app.dragging = true;
            break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (e.button.button == SDL_BUTTON_LEFT) app.dragging = false;
            break;
        case SDL_EVENT_MOUSE_MOTION:
            if (app.dragging && !io.WantCaptureMouse) {
                app.cx.subDouble(e.motion.xrel * app.view.scale);
                app.cy.addDouble(e.motion.yrel * app.view.scale);
                app.dirty = true;
            }
            break;
        case SDL_EVENT_MOUSE_WHEEL: {
            if (io.WantCaptureMouse) break;
            float mx, my;
            SDL_GetMouseState(&mx, &my);
            const double factor = std::pow(0.8, (double)e.wheel.y);
            zoomAt(app, mx, my, factor);
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
        const int digits = std::max(5, (int)std::ceil(-std::log10(app.view.scale)) + 3);
        ImGui::Text("center  %s", app.cx.toString(digits).c_str());
        ImGui::Text("        %s i", app.cy.toString(digits).c_str());
        const double zoom = 3.2 / (app.view.scale * app.view.width);
        ImGui::Text("zoom    %.3g x   (%ld bits)", zoom, (long)app.cx.prec());
        ImGui::Text("size    %d x %d", app.view.width, app.view.height);
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
        ImGui::Text("shade   %.2f ms", app.renderer.lastShadeMs());
        ImGui::Text("frame   %.0f fps", app.fps);
        ImGui::TextDisabled("drag: pan  wheel: zoom  R: reset  Tab: hide");

        if (ImGui::CollapsingHeader("Shading")) {
            ShadeParams& sp = app.shade;
            bool ch = false;
            ch |= ImGui::Checkbox("animate", &app.animate);
            ImGui::SameLine();
            ch |= ImGui::SliderFloat("speed", &app.animSpeed, -1.f, 1.f, "%.3f cyc/s");
            ch |= ImGui::SliderFloat("density", &sp.density, 4.f, 1024.f, "%.1f",
                                     ImGuiSliderFlags_Logarithmic);
            ch |= ImGui::SliderFloat("offset", &sp.offset, 0.f, 1.f);
            bool logScale = sp.logScale != 0;
            if (ImGui::Checkbox("log scale", &logScale)) { sp.logScale = logScale; ch = true; }
            ch |= ImGui::SliderFloat("edge (DE)", &sp.deStrength, 0.f, 4.f);
            ch |= ImGui::SliderFloat("exposure", &sp.exposure, 0.f, 4.f);
            ch |= ImGui::ColorEdit3("inside", sp.inside, ImGuiColorEditFlags_Float);
            ImGui::TextDisabled("palette a + b cos(2pi(c t + d))");
            ch |= ImGui::DragFloat3("a", sp.a, 0.01f, -1.f, 2.f);
            ch |= ImGui::DragFloat3("b", sp.b, 0.01f, -1.f, 2.f);
            ch |= ImGui::DragFloat3("c", sp.c, 0.01f, -4.f, 4.f);
            ch |= ImGui::DragFloat3("d", sp.d, 0.01f, -1.f, 1.f);
            if (ImGui::Button("reset palette")) { sp = ShadeParams(); ch = true; }
            if (ch) app.shadeDirty = true;
        }
    }
    ImGui::End();
}

}  // namespace

int main(int argc, char** argv) {
    SDL_SetMainReady();
    // Optional start location: mandelgpu <re> <im> <scale> [maxIter]
    // Numbers are decimal strings, any length. scale is complex units per pixel.
    std::string argRe, argIm;
    double argScale = 0.0;
    int argIter = 0;
    if (argc >= 4) {
        argRe = argv[1];
        argIm = argv[2];
        argScale = std::atof(argv[3]);
        if (argc >= 5) argIter = std::atoi(argv[4]);
    }
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

    Uint64 lastTick = SDL_GetPerformanceCounter();
    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (!handleEvent(app, e)) running = false;
        }

        int pw = 0, ph = 0;
        SDL_GetWindowSizeInPixels(app.window, &pw, &ph);
        if (app.display.resize(pw, ph)) {
            const bool first = app.view.width == 0;
            app.view.width = pw;
            app.view.height = ph;
            if (first) {
                resetView(app);
                if (argScale > 0.0) {
                    app.view.scale = argScale;
                    updatePrecision(app);
                    app.cx.set(argRe);
                    app.cy.set(argIm);
                    if (argIter > 0) app.view.maxIter = argIter;
                }
            }
            app.renderer.bindPixelBuffer(app.display.pbo(), pw, ph);
            app.shown.valid = false;
            app.dirty = true;
        }

        if (app.dirty && pw > 0 && ph > 0) {
            rebuildReference(app);
            app.renderer.beginIterate(app.view);
            app.showingNewPass = false;
            app.dirty = false;
        }
        // One slice in flight at a time. Shade only when the stream is idle so
        // the display never queues behind a slice.
        if (!app.renderer.iterateBusy()) {
            const bool sliceDone = app.renderer.takeSliceFinished();
            if (!app.showingNewPass && newPassWorthShowing(app)) {
                app.showingNewPass = true;
                app.shadeDirty = true;
            }
            if (app.showingNewPass) {
                if (sliceDone) app.shadeDirty = true;
                if (app.animate) app.shadeDirty = true;
                if (app.shadeDirty && pw > 0 && ph > 0) {
                    const int fill = app.renderer.iterateDone() ? 1 : app.renderer.completedStride();
                    if (app.renderer.shade(app.shade, app.animTime, app.view.scale, fill)) {
                        app.display.upload();
                        markShown(app);
                    }
                    app.shadeDirty = false;
                }
            }
            if (!app.renderer.iterateDone()) app.renderer.stepIterate();
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        {
            float u0, v0, u1, v1;
            reprojectUv(app, u0, v0, u1, v1);
            app.display.draw(u0, v0, u1, v1);
        }
        drawUi(app);
        ImGui::Render();

        glViewport(0, 0, pw, ph);
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(app.window);

        const Uint64 now = SDL_GetPerformanceCounter();
        const double dt = (double)(now - lastTick) / (double)SDL_GetPerformanceFrequency();
        lastTick = now;
        if (dt > 0) app.fps = 0.9 * app.fps + 0.1 * (1.0 / dt);
        if (app.animate) app.animTime += (float)dt * app.animSpeed;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DestroyContext(app.gl);
    SDL_DestroyWindow(app.window);
    SDL_Quit();
    return 0;
}
