#pragma once
// Shading + post-processing presets: built-in ones and files the user saves.
#include <string>
#include <vector>
#include "field.h"

struct Preset {
    ShadeParams shade;
    PostParams post;
    bool animate = false;
};

enum { BUILTIN_PRESET_COUNT = 9 };
extern const char* const kBuiltinPresetNames[BUILTIN_PRESET_COUNT];
Preset builtinPreset(int which);

// Saved presets live in the per-user preferences folder as plain text.
std::string presetDir();
std::vector<std::string> listSavedPresets();
bool savePreset(const std::string& name, const Preset& p);
bool loadPreset(const std::string& name, Preset& p);
bool deletePreset(const std::string& name);
// Same format, explicit path (screenshot sidecars, --loadfile).
bool savePresetFile(const std::string& path, const Preset& p);
bool loadPresetFile(const std::string& path, Preset& p);
