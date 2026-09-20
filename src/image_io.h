#pragma once
#include <cstdint>
// Write an RGBA8 buffer (top row first) as PNG. Returns false on failure.
bool writePng(const char* path, const uint32_t* rgba, int width, int height);

// Write linear RGB floats (3 per pixel, top row first) as a half-float EXR.
bool writeExr(const char* path, const float* rgb, int width, int height);
