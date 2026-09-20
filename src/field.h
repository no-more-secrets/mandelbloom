#pragma once
// Shared between host C++ and CUDA. Plain structs only.

// One sample of the iteration field. Written by the iteration kernel when a
// pixel finishes, read by the shading kernel. Coloring never touches
// iteration state. 24 bytes.
//
// gen is the render generation that produced the sample (0 = never
// computed). The field is not cleared when the view changes: samples from
// older generations stay until overwritten, and the display maps them
// through the view they were rendered with.
struct FieldSample {
    float iter;   // smooth escape iteration; < 0 means inside the set
    float de;     // distance estimate in pixels of the render field (0 if inside)
    float angle;  // arg(z_final) in [-pi, pi]
    float nx, ny; // exterior surface normal (Milnor: z / dz, normalised), 0 inside
    float gen;    // generation id, 0 = none
};

#define MAX_STOPS 8

enum ShadeMode {
    SHADE_SMOOTH = 0,   // palette over the smooth iteration count
    SHADE_LOGSTEPS,     // brightness ramps up within each band (KF "log steps")
    SHADE_WAVE,         // dark-light sine wave over the palette
    SHADE_PANELS,       // flat panels with dark gaps
    SHADE_ANGLE,        // palette over the escape angle
    SHADE_DISTANCE,     // palette over the log of the distance estimate
    SHADE_MODE_COUNT
};

// Everything the shading pass needs. Changing these never re-iterates.
struct ShadeParams {
    int mode = SHADE_SMOOTH;

    // Palette position t (in cycles) comes from the iteration count:
    //   logScale: t = log2(iter) * density / 64,  else t = iter / density
    float density = 64.f;
    float offset = 0.f;      // phase offset in cycles
    int logScale = 1;
    float special = 4.f;     // per-mode parameter (steps/waves/panels per cycle, DE falloff)

    // Palette: gradient stops blended in OKLab, or a cosine palette.
    int paletteType = 0;     // 0 gradient, 1 cosine
    int stopCount = 5;
    float stopPos[MAX_STOPS] = {0.f, 0.25f, 0.5f, 0.75f, 1.f, 0, 0, 0};
    float stopColor[MAX_STOPS][3] = {{0.02f, 0.02f, 0.10f}, {0.10f, 0.40f, 0.85f},
                                     {0.95f, 0.95f, 0.95f}, {0.95f, 0.60f, 0.05f},
                                     {0.02f, 0.02f, 0.10f}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    float a[3] = {0.5f, 0.5f, 0.5f};  // cosine: a + b cos(2 pi (c t + d))
    float b[3] = {0.5f, 0.5f, 0.5f};
    float c[3] = {1.0f, 1.0f, 1.0f};
    float d[3] = {0.00f, 0.33f, 0.67f};

    float deStrength = 1.f;  // 0 disables distance-estimate edge darkening
    float exposure = 1.f;
    float inside[3] = {0.f, 0.f, 0.f};

    // Slope lighting from the exterior normal (pseudo-3D relief).
    int slopes = 0;
    float slopeAngle = 45.f;    // light direction, degrees, 0 = from the right
    float slopeHeight = 1.5f;   // light elevation; higher = flatter relief
    float slopeStrength = 0.7f; // 0..1, how dark the shadow side gets

    // Thin antialiased lines on iteration band boundaries.
    int lines = 0;
    float lineDensity = 1.f;    // lines per iteration (fractions allowed)
    float lineWidth = 1.2f;     // pixels
    float lineStrength = 0.85f; // 0..1 blend toward lineColor
    float lineColor[3] = {0.f, 0.f, 0.f};

    // Animation, driven by the time the shader receives (seconds).
    float cycleSpeed = 0.05f;   // palette cycles per second
    float lightSpeed = 0.f;     // light angle, degrees per second
    float waveSpeed = 0.f;      // wave/panel phase, cycles per second
};
