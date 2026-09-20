#pragma once
// Shared between host C++ and CUDA. Plain structs only.

// One sample of the iteration field. Written by the iteration kernel,
// read by the shading kernel. Coloring never touches iteration state.
struct FieldSample {
    float iter;   // smooth escape iteration; < 0 means inside the set
    float de;     // distance estimate in complex units (0 if inside)
    float angle;  // arg(z_final) in [-pi, pi]
    float pad;
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
    int logScale = 1;       // 1: t = log2(iter) * density/8, 0: t = iter / density
    float deStrength = 1.f; // 0 disables distance-estimate edge darkening
    float exposure = 1.f;
    float inside[3] = {0.f, 0.f, 0.f};
};
