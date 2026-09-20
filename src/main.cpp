#include <glad/glad.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_opengl3.h>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include "gl_display.h"
#include "render_cuda.h"

namespace {

struct App {
    SDL_Window* window = nullptr;
    SDL_GLContext gl = nullptr;
    GlDisplay display;
    CudaRenderer renderer;
    ViewParams view;
    bool dirty = true;
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

void zoomAt(App& app, float mx, float my, double factor) {
    // Keep the complex point under the cursor fixed while scaling.
    const double hx = 0.5 * app.view.width, hy = 0.5 * app.view.height;
    const double ur = app.view.cx + (mx - hx) * app.view.scale;
    const double ui = app.view.cy - (my - hy) * app.view.scale;
    app.view.scale *= factor;
    app.view.cx = ur - (mx - hx) * app.view.scale;
    app.view.cy = ui + (my - hy) * app.view.scale;
    app.dirty = true;
}

void resetView(App& app) {
    app.view.cx = -0.5;
    app.view.cy = 0.0;
    app.view.scale = 3.2 / std::max(1, app.view.width);
    app.dirty = true;
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
                app.view.cx -= e.motion.xrel * app.view.scale;
                app.view.cy += e.motion.yrel * app.view.scale;
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
        ImGui::Text("center  %.17g", app.view.cx);
        ImGui::Text("        %.17g i", app.view.cy);
        const double zoom = 3.2 / (app.view.scale * app.view.width);
        ImGui::Text("zoom    %.3g x", zoom);
        ImGui::Text("size    %d x %d", app.view.width, app.view.height);
        if (ImGui::SliderInt("max iter", &app.view.maxIter, 64, 65536, "%d",
                             ImGuiSliderFlags_Logarithmic)) {
            app.dirty = true;
        }
        ImGui::Separator();
        ImGui::Text("render  %.2f ms", app.renderer.lastRenderMs());
        ImGui::Text("frame   %.0f fps", app.fps);
        ImGui::TextDisabled("drag: pan  wheel: zoom  R: reset  Tab: hide");
    }
    ImGui::End();
}

}  // namespace

int main(int, char**) {
    SDL_SetMainReady();
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
            if (first) resetView(app);
            app.renderer.bindPixelBuffer(app.display.pbo(), pw, ph);
            app.dirty = true;
        }

        if (app.dirty && pw > 0 && ph > 0) {
            if (app.renderer.render(app.view)) app.display.upload();
            app.dirty = false;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        app.display.draw();
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
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DestroyContext(app.gl);
    SDL_DestroyWindow(app.window);
    SDL_Quit();
    return 0;
}
