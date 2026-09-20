#pragma once
#include <string>
#include <vector>

// Scripted input for testing without touching the real mouse. Commands,
// separated by ';':
//   wait:MS            idle for MS milliseconds
//   wheel:N            N wheel notches (negative = zoom out), one per frame
//   wheelgap:MS        delay between notches (default 60)
//   drag:DX,DY,STEPS   left-drag by DX,DY pixels over STEPS frames
//   pan:DX,DY          pan by DX,DY pixels directly (no drag, no inertia)
//   shot:PATH          save the current frame as PNG
//   quit               exit the app
// Wheel and drag events are posted through SDL so the normal handlers run.
struct ScriptStep {
    std::string cmd;
    std::string arg;
};

class Script {
public:
    bool parse(const std::string& text);
    bool active() const { return !steps_.empty() && pos_ < steps_.size(); }
    // Called once per frame with the current time; returns a screenshot path
    // to capture this frame (empty if none) and sets quit when done.
    std::string tick(double nowSec, int winW, int winH, bool& quit);
    // Set when the last tick issued a direct pan; consumed by the caller.
    bool takePan(int& dx, int& dy) {
        if (!panPending_) return false;
        dx = panDx_; dy = panDy_; panPending_ = false; return true;
    }

private:
    std::vector<ScriptStep> steps_;
    size_t pos_ = 0;
    double waitUntil_ = 0.0;
    int wheelLeft_ = 0;
    int wheelGapMs_ = 60;
    int dragStepsLeft_ = 0;
    double dragDx_ = 0, dragDy_ = 0;
    double dragAccX_ = 0, dragAccY_ = 0;
    double mouseX_ = 0, mouseY_ = 0;
    bool started_ = false;
    bool panPending_ = false;
    int panDx_ = 0, panDy_ = 0;
};
