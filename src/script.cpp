#include "script.h"
#include <SDL3/SDL.h>
#include <cstdlib>
#include <cstring>

namespace {

void pushWheel(double x, double y, float notches) {
    SDL_Event e{};
    e.type = SDL_EVENT_MOUSE_WHEEL;
    e.wheel.timestamp = SDL_GetTicksNS();
    e.wheel.x = 0.f;
    e.wheel.y = notches;
    e.wheel.mouse_x = (float)x;
    e.wheel.mouse_y = (float)y;
    e.wheel.direction = SDL_MOUSEWHEEL_NORMAL;
    SDL_PushEvent(&e);
}

void pushButton(double x, double y, bool down) {
    SDL_Event e{};
    e.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
    e.button.timestamp = SDL_GetTicksNS();
    e.button.button = SDL_BUTTON_LEFT;
    e.button.down = down;
    e.button.clicks = 1;
    e.button.x = (float)x;
    e.button.y = (float)y;
    SDL_PushEvent(&e);
}

void pushMotion(double x, double y, double dx, double dy) {
    SDL_Event e{};
    e.type = SDL_EVENT_MOUSE_MOTION;
    e.motion.timestamp = SDL_GetTicksNS();
    e.motion.state = SDL_BUTTON_LMASK;
    e.motion.x = (float)x;
    e.motion.y = (float)y;
    e.motion.xrel = (float)dx;
    e.motion.yrel = (float)dy;
    SDL_PushEvent(&e);
}

}  // namespace

bool Script::parse(const std::string& text) {
    steps_.clear();
    size_t start = 0;
    while (start <= text.size()) {
        size_t end = text.find(';', start);
        if (end == std::string::npos) end = text.size();
        std::string item = text.substr(start, end - start);
        while (!item.empty() && (item.back() == ' ' || item.back() == '\r' || item.back() == '\n'))
            item.pop_back();
        while (!item.empty() && item.front() == ' ') item.erase(item.begin());
        if (!item.empty()) {
            ScriptStep s;
            const size_t colon = item.find(':');
            s.cmd = item.substr(0, colon);
            if (colon != std::string::npos) s.arg = item.substr(colon + 1);
            steps_.push_back(s);
        }
        start = end + 1;
    }
    pos_ = 0;
    return !steps_.empty();
}

std::string Script::tick(double now, int winW, int winH, bool& quit) {
    quit = false;
    if (!started_) {
        started_ = true;
        mouseX_ = 0.5 * winW + 0.1 * winW;  // off-centre so zooms visibly shift
        mouseY_ = 0.5 * winH + 0.1 * winH;
    }
    // Ongoing multi-frame actions.
    if (now < waitUntil_) return "";
    if (wheelLeft_ != 0) {
        const float dir = wheelLeft_ > 0 ? 1.f : -1.f;
        pushWheel(mouseX_, mouseY_, dir);
        wheelLeft_ -= (int)dir;
        waitUntil_ = now + wheelGapMs_ * 0.001;
        return "";
    }
    if (dragStepsLeft_ > 0) {
        dragAccX_ += dragDx_;
        dragAccY_ += dragDy_;
        const int ix = (int)dragAccX_, iy = (int)dragAccY_;
        dragAccX_ -= ix;
        dragAccY_ -= iy;
        mouseX_ += ix;
        mouseY_ += iy;
        pushMotion(mouseX_, mouseY_, ix, iy);
        if (--dragStepsLeft_ == 0) pushButton(mouseX_, mouseY_, false);
        waitUntil_ = now + 0.016;
        return "";
    }
    while (pos_ < steps_.size()) {
        const ScriptStep& s = steps_[pos_++];
        if (s.cmd == "wait") {
            waitUntil_ = now + std::atoi(s.arg.c_str()) * 0.001;
            return "";
        } else if (s.cmd == "wheelgap") {
            wheelGapMs_ = std::atoi(s.arg.c_str());
        } else if (s.cmd == "wheel") {
            wheelLeft_ = std::atoi(s.arg.c_str());
            return "";
        } else if (s.cmd == "drag") {
            int dx = 0, dy = 0, steps = 1;
            std::sscanf(s.arg.c_str(), "%d,%d,%d", &dx, &dy, &steps);
            if (steps < 1) steps = 1;
            dragDx_ = (double)dx / steps;
            dragDy_ = (double)dy / steps;
            dragAccX_ = dragAccY_ = 0;
            dragStepsLeft_ = steps;
            pushButton(mouseX_, mouseY_, true);
            return "";
        } else if (s.cmd == "pan") {
            int dx = 0, dy = 0;
            std::sscanf(s.arg.c_str(), "%d,%d", &dx, &dy);
            panDx_ = dx;
            panDy_ = dy;
            panPending_ = true;
            waitUntil_ = now + 0.016;
            return "";
        } else if (s.cmd == "screenshot") {
            shotReq_ = std::atoi(s.arg.c_str()) >= 2 ? 2 : 1;
            waitUntil_ = now + 0.016;
            return "";
        } else if (s.cmd == "animate") {
            animate_ = std::atoi(s.arg.c_str()) ? 1 : 0;
            waitUntil_ = now + 0.016;
            return "";
        } else if (s.cmd == "shot") {
            return s.arg;
        } else if (s.cmd == "quit") {
            quit = true;
            return "";
        }
    }
    return "";
}
