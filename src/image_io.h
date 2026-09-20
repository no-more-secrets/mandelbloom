#pragma once
#include <cstdint>
// Write an RGBA8 buffer (top row first) as PNG. Returns false on failure.
bool writePng(const char* path, const uint32_t* rgba, int width, int height);
