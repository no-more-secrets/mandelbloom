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
    float de;     // distance estimate in complex units (0 if inside)
    float angle;  // arg(z_final) in [-pi, pi]
    float nx, ny; // exterior surface normal (Milnor: z / dz, normalised), 0 inside
    float gen;    // generation id, 0 = none
};

// Everything the shading pass needs. Changing these never re-iterates.
struct ShadeParams {
    // Cosine palette: color(t) = a + b * cos(2*pi*(c*t + d))
    float a[3] = {0.5f, 0.5f, 0.5f};
    float b[3] = {0.5f, 0.5f, 0.5f};
    float c[3] = {1.0f, 1.0f, 1.0f};
    float d[3] = {0.00f, 0.33f, 0.67f};
    float density = 64.f;   // iterations per palette cycle (linear mode)
    float offset = 0.f;     // palette phase offset in cycles
    int logScale = 1;       // 1: t = log2(iter) * density/64, 0: t = iter / density
    float deStrength = 1.f; // 0 disables distance-estimate edge darkening
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
};
