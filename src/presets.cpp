#include "presets.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

namespace fs = std::filesystem;

const char* const kBuiltinPresetNames[BUILTIN_PRESET_COUNT] = {
    "classic", "pastel lines", "relief", "mono lines", "ocean wave",
    "ember panels", "angle", "distance", "solar"};

namespace {

void setStops(ShadeParams& sp, int n, const float* pos, const float (*rgb)[3]) {
    sp.paletteType = 0;
    sp.stopCount = n;
    for (int i = 0; i < n; ++i) {
        sp.stopPos[i] = pos[i];
        for (int k = 0; k < 3; ++k) sp.stopColor[i][k] = rgb[i][k];
    }
}

// Field table for the text format: one "name value(s)" line per entry.
struct Field {
    const char* name;
    enum { F, I } type;
    void* ptr;
    int count;
};

std::vector<Field> fields(Preset& p) {
    ShadeParams& s = p.shade;
    PostParams& q = p.post;
    return {
        {"mode", Field::I, &s.mode, 1},
        {"density", Field::F, &s.density, 1},
        {"offset", Field::F, &s.offset, 1},
        {"logScale", Field::I, &s.logScale, 1},
        {"special", Field::F, &s.special, 1},
        {"paletteType", Field::I, &s.paletteType, 1},
        {"stopCount", Field::I, &s.stopCount, 1},
        {"stopPos", Field::F, s.stopPos, MAX_STOPS},
        {"stopColor", Field::F, &s.stopColor[0][0], MAX_STOPS * 3},
        {"cosA", Field::F, s.a, 3},
        {"cosB", Field::F, s.b, 3},
        {"cosC", Field::F, s.c, 3},
        {"cosD", Field::F, s.d, 3},
        {"deStrength", Field::F, &s.deStrength, 1},
        {"exposure", Field::F, &s.exposure, 1},
        {"inside", Field::F, s.inside, 3},
        {"slopes", Field::I, &s.slopes, 1},
        {"slopeAngle", Field::F, &s.slopeAngle, 1},
        {"slopeHeight", Field::F, &s.slopeHeight, 1},
        {"slopeStrength", Field::F, &s.slopeStrength, 1},
        {"lines", Field::I, &s.lines, 1},
        {"lineDensity", Field::F, &s.lineDensity, 1},
        {"lineWidth", Field::F, &s.lineWidth, 1},
        {"lineStrength", Field::F, &s.lineStrength, 1},
        {"lineColor", Field::F, s.lineColor, 3},
        {"wavePhase", Field::F, &s.wavePhase, 1},
        {"cycleSpeed", Field::F, &s.cycleSpeed, 1},
        {"lightSpeed", Field::F, &s.lightSpeed, 1},
        {"waveSpeed", Field::F, &s.waveSpeed, 1},
        {"tonemap", Field::I, &q.tonemap, 1},
        {"hdrBoost", Field::F, &q.hdrBoost, 1},
        {"bloomIntensity", Field::F, &q.bloomIntensity, 1},
        {"bloomThreshold", Field::F, &q.bloomThreshold, 1},
        {"bloomKnee", Field::F, &q.bloomKnee, 1},
        {"bloomRadius", Field::I, &q.bloomRadius, 1},
        {"bloomWide", Field::F, &q.bloomWide, 1},
        {"vignette", Field::F, &q.vignette, 1},
        {"vignetteSoft", Field::F, &q.vignetteSoft, 1},
        {"aberration", Field::F, &q.aberration, 1},
        {"grain", Field::F, &q.grain, 1},
        {"saturation", Field::F, &q.saturation, 1},
        {"contrast", Field::F, &q.contrast, 1},
        {"gain", Field::F, &q.gain, 1},
        {"sharpen", Field::F, &q.sharpen, 1},
    };
}

bool validName(const std::string& n) {
    if (n.empty() || n.size() > 64) return false;
    for (char c : n)
        if (!(std::isalnum((unsigned char)c) || c == ' ' || c == '-' || c == '_' || c == '.'))
            return false;
    return true;
}

}  // namespace

Preset builtinPreset(int which) {
    Preset p;
    ShadeParams& sp = p.shade;
    switch (which) {
        case 1: {  // Pastel lines
            const float pos[5] = {0.f, 0.25f, 0.5f, 0.75f, 1.f};
            const float rgb[5][3] = {{0.98f, 0.80f, 0.86f}, {0.72f, 0.86f, 0.98f},
                                     {0.99f, 0.96f, 0.72f}, {0.86f, 0.74f, 0.96f},
                                     {0.98f, 0.80f, 0.86f}};
            setStops(sp, 5, pos, rgb);
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
        case 2: {  // Relief
            sp.paletteType = 1;
            sp.slopes = 1;
            sp.slopeAngle = 30.f;
            sp.slopeHeight = 1.0f;
            sp.slopeStrength = 0.85f;
            sp.deStrength = 0.f;
            break;
        }
        case 3: {  // Mono lines
            const float pos[2] = {0.f, 1.f};
            const float rgb[2][3] = {{0.96f, 0.96f, 0.96f}, {0.90f, 0.90f, 0.90f}};
            setStops(sp, 2, pos, rgb);
            sp.slopes = 1;
            sp.slopeStrength = 0.35f;
            sp.lines = 1;
            sp.lineWidth = 1.0f;
            sp.lineStrength = 0.9f;
            sp.deStrength = 0.4f;
            break;
        }
        case 4: {  // Ocean wave
            const float pos[4] = {0.f, 0.35f, 0.7f, 1.f};
            const float rgb[4][3] = {{0.02f, 0.05f, 0.20f}, {0.05f, 0.45f, 0.75f},
                                     {0.75f, 0.95f, 0.95f}, {0.02f, 0.05f, 0.20f}};
            setStops(sp, 4, pos, rgb);
            sp.mode = SHADE_WAVE;
            sp.special = 3.f;
            sp.waveSpeed = 0.15f;
            sp.slopes = 1;
            sp.slopeStrength = 0.4f;
            break;
        }
        case 5: {  // Ember panels
            const float pos[5] = {0.f, 0.3f, 0.55f, 0.8f, 1.f};
            const float rgb[5][3] = {{0.05f, 0.0f, 0.0f}, {0.6f, 0.05f, 0.0f},
                                     {1.0f, 0.55f, 0.05f}, {1.0f, 0.95f, 0.6f},
                                     {0.05f, 0.0f, 0.0f}};
            setStops(sp, 5, pos, rgb);
            sp.mode = SHADE_PANELS;
            sp.special = 4.f;
            sp.density = 48.f;
            sp.deStrength = 0.5f;
            break;
        }
        case 6: {  // Angle
            sp.paletteType = 1;
            sp.mode = SHADE_ANGLE;
            sp.special = 1.f;
            sp.slopes = 1;
            sp.slopeStrength = 0.5f;
            sp.deStrength = 0.8f;
            break;
        }
        case 7: {  // Distance
            const float pos[3] = {0.f, 0.5f, 1.f};
            const float rgb[3][3] = {{0.05f, 0.05f, 0.08f}, {0.55f, 0.55f, 0.6f},
                                     {0.98f, 0.98f, 1.0f}};
            setStops(sp, 3, pos, rgb);
            sp.mode = SHADE_DISTANCE;
            sp.special = 3.f;
            sp.deStrength = 0.f;
            sp.cycleSpeed = 0.f;
            break;
        }
        case 8: {  // Solar: Sam's HDR look, 2026-09-20
            const float pos[5] = {0.f, 0.38f, 0.55f, 0.58f, 1.f};
            const float rgb[5][3] = {{0.06f, 0.07f, 0.22f}, {0.35f, 0.55f, 0.85f},
                                     {1.0f, 1.0f, 1.0f}, {1.0f, 0.60f, 0.10f},
                                     {0.06f, 0.07f, 0.22f}};
            setStops(sp, 5, pos, rgb);
            sp.density = 107.8f;
            sp.slopes = 1;
            sp.slopeAngle = 30.f;
            sp.slopeHeight = 1.0f;
            sp.slopeStrength = 0.85f;
            sp.deStrength = 0.899f;
            sp.exposure = 0.852f;
            sp.cycleSpeed = -0.42f;
            p.animate = true;
            PostParams& q = p.post;
            q.tonemap = 2;
            q.hdrBoost = 1.41f;
            q.bloomIntensity = 0.906f;
            q.bloomThreshold = 0.815f;
            q.bloomKnee = 0.3f;
            q.bloomRadius = 9;
            q.bloomWide = 1.644f;
            q.vignette = 0.926f;
            q.vignetteSoft = 0.546f;
            q.aberration = 3.705f;
            q.saturation = 1.409f;
            q.contrast = 1.287f;
            q.gain = 1.832f;
            break;
        }
        default:
            break;
    }
    return p;
}

std::string presetDir() {
    char* pref = SDL_GetPrefPath("NMS", "Mandelbloom");  // APPDATA/NMS/Mandelbloom/
    std::string dir = pref ? pref : "";
    if (pref) SDL_free(pref);
    dir += "presets";
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir;
}

std::vector<std::string> listSavedPresets() {
    std::vector<std::string> names;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(presetDir(), ec)) {
        if (e.path().extension() == ".preset") names.push_back(e.path().stem().string());
    }
    std::sort(names.begin(), names.end());
    return names;
}

bool savePreset(const std::string& name, const Preset& pIn) {
    if (!validName(name)) return false;
    return savePresetFile((fs::path(presetDir()) / (name + ".preset")).string(), pIn);
}

bool savePresetFile(const std::string& path, const Preset& pIn) {
    Preset p = pIn;
    std::ofstream out(path);
    if (!out) return false;
    out << "mandelgpu-preset 1\n";
    out << "animate " << (p.animate ? 1 : 0) << "\n";
    for (const Field& f : fields(p)) {
        out << f.name;
        for (int i = 0; i < f.count; ++i) {
            if (f.type == Field::F) out << ' ' << ((float*)f.ptr)[i];
            else out << ' ' << ((int*)f.ptr)[i];
        }
        out << "\n";
    }
    return true;
}

bool loadPreset(const std::string& name, Preset& p) {
    return loadPresetFile((fs::path(presetDir()) / (name + ".preset")).string(), p);
}

bool loadPresetFile(const std::string& path, Preset& p) {
    std::ifstream in(path);
    if (!in) return false;
    std::string header;
    std::getline(in, header);
    if (header.rfind("mandelgpu-preset", 0) != 0) return false;
    p = Preset();
    std::map<std::string, Field> byName;
    for (const Field& f : fields(p)) byName.emplace(f.name, f);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key)) continue;
        if (key == "animate") {
            int v = 0;
            ss >> v;
            p.animate = v != 0;
            continue;
        }
        auto it = byName.find(key);
        if (it == byName.end()) continue;
        const Field& f = it->second;
        for (int i = 0; i < f.count; ++i) {
            if (f.type == Field::F) {
                float v;
                if (ss >> v) ((float*)f.ptr)[i] = v;
            } else {
                int v;
                if (ss >> v) ((int*)f.ptr)[i] = v;
            }
        }
    }
    return true;
}

bool deletePreset(const std::string& name) {
    if (!validName(name)) return false;
    std::error_code ec;
    return fs::remove(fs::path(presetDir()) / (name + ".preset"), ec);
}
