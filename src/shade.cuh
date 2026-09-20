#pragma once
// Device-side coloring. Included by render_cuda.cu only.
#include <cstdint>
#include "field.h"

__device__ __forceinline__ float smoothstepf(float e0, float e1, float x) {
    const float t = fminf(fmaxf((x - e0) / (e1 - e0), 0.f), 1.f);
    return t * t * (3.f - 2.f * t);
}

__device__ __forceinline__ float fracf(float x) { return x - floorf(x); }

// ---- OKLab (Bjorn Ottosson), for perceptually even gradient blending ----
__device__ __forceinline__ float3 linearToOklab(float3 c) {
    const float l = 0.4122214708f * c.x + 0.5363325363f * c.y + 0.0514459929f * c.z;
    const float m = 0.2119034982f * c.x + 0.6806995451f * c.y + 0.1073969566f * c.z;
    const float s = 0.0883024619f * c.x + 0.2817188376f * c.y + 0.6299787005f * c.z;
    const float l_ = cbrtf(l), m_ = cbrtf(m), s_ = cbrtf(s);
    return make_float3(0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
                       1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
                       0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_);
}

__device__ __forceinline__ float3 oklabToLinear(float3 c) {
    const float l_ = c.x + 0.3963377774f * c.y + 0.2158037573f * c.z;
    const float m_ = c.x - 0.1055613458f * c.y - 0.0638541728f * c.z;
    const float s_ = c.x - 0.0894841775f * c.y - 1.2914855480f * c.z;
    const float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    return make_float3(+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
                       -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
                       -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s);
}

__device__ __forceinline__ float3 srgbToLinear(float3 c) {
    auto f = [](float v) { return v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f); };
    return make_float3(f(c.x), f(c.y), f(c.z));
}

// Gradient stops are edited in sRGB; blend in OKLab; return linear RGB.
__device__ __forceinline__ float3 gradientPalette(const ShadeParams& p, float t) {
    const float u = fracf(t);
    const int n = p.stopCount < 2 ? 2 : (p.stopCount > MAX_STOPS ? MAX_STOPS : p.stopCount);
    int i = 0;
    while (i + 1 < n - 1 && u >= p.stopPos[i + 1]) ++i;
    const float p0 = p.stopPos[i], p1 = p.stopPos[i + 1];
    const float f = p1 > p0 ? fminf(fmaxf((u - p0) / (p1 - p0), 0.f), 1.f) : 0.f;
    const float3 c0 = linearToOklab(srgbToLinear(
        make_float3(p.stopColor[i][0], p.stopColor[i][1], p.stopColor[i][2])));
    const float3 c1 = linearToOklab(srgbToLinear(
        make_float3(p.stopColor[i + 1][0], p.stopColor[i + 1][1], p.stopColor[i + 1][2])));
    const float3 m = make_float3(c0.x + (c1.x - c0.x) * f, c0.y + (c1.y - c0.y) * f,
                                 c0.z + (c1.z - c0.z) * f);
    const float3 lin = oklabToLinear(m);
    return make_float3(fmaxf(lin.x, 0.f), fmaxf(lin.y, 0.f), fmaxf(lin.z, 0.f));
}

__device__ __forceinline__ float3 cosPalette(const ShadeParams& p, float t) {
    const float k = 6.2831853f;
    return make_float3(p.a[0] + p.b[0] * cosf(k * (p.c[0] * t + p.d[0])),
                       p.a[1] + p.b[1] * cosf(k * (p.c[1] * t + p.d[1])),
                       p.a[2] + p.b[2] * cosf(k * (p.c[2] * t + p.d[2])));
}

__device__ __forceinline__ float3 palette(const ShadeParams& p, float t) {
    return p.paletteType == 1 ? cosPalette(p, t) : gradientPalette(p, t);
}

// Screen-space rate of change of the iteration count, per display pixel,
// used to draw lines at constant width. Zero when unknown.
struct IterGradient {
    float dx = 0.f, dy = 0.f;
    bool valid = false;
};

// Shade one sample to linear RGB. de is in pixels; pixelScale is unused
// now but kept for future colourings in complex units.
__device__ __forceinline__ float3 shadeSample(const FieldSample& s, const ShadeParams& p,
                                              float timeSec, float pixelScale,
                                              const IterGradient& g) {
    if (s.iter < 0.f || s.gen < 0.5f) return make_float3(p.inside[0], p.inside[1], p.inside[2]);

    // Palette position in cycles.
    float t;
    if (p.logScale) {
        t = log2f(fmaxf(s.iter, 1.f)) * (p.density / 64.f);
    } else {
        t = s.iter / p.density;
    }
    t += p.offset + p.cycleSpeed * timeSec;
    const float phase = p.waveSpeed * timeSec;

    float3 col;
    switch (p.mode) {
        case SHADE_LOGSTEPS: {
            // Within each of `special` bands per cycle, brightness ramps
            // logarithmically from dark to full.
            const float f = fracf(t * p.special + phase);
            const float ramp = logf(1.f + f * 1.7182818f);  // 0..1
            const float3 c = palette(p, t);
            const float k = 0.25f + 0.75f * ramp;
            col = make_float3(c.x * k, c.y * k, c.z * k);
            break;
        }
        case SHADE_WAVE: {
            const float w = 0.5f + 0.5f * sinf(6.2831853f * (t * p.special + phase));
            const float3 c = palette(p, t);
            col = make_float3(c.x * w, c.y * w, c.z * w);
            break;
        }
        case SHADE_PANELS: {
            // Flat panels with soft dark gaps of ~15% of the panel.
            const float f = fracf(t * p.special + phase);
            const float gap = 0.15f;
            const float k = smoothstepf(0.f, gap, f) * smoothstepf(1.f, 1.f - gap, f);
            const float3 c = palette(p, t);
            const float m = 0.1f + 0.9f * k;
            col = make_float3(c.x * m, c.y * m, c.z * m);
            break;
        }
        case SHADE_ANGLE: {
            // Escape angle over the palette; `special` cycles per turn.
            const float ang = s.angle * 0.15915494f;  // turns
            col = palette(p, ang * p.special + p.offset + p.cycleSpeed * timeSec);
            break;
        }
        case SHADE_DISTANCE: {
            // Log distance from the set, in pixels; `special` decades per cycle.
            const float dpx = fmaxf(s.de, 1e-6f);
            const float v = log10f(dpx) / fmaxf(p.special, 0.01f);
            col = palette(p, v + p.offset + p.cycleSpeed * timeSec);
            break;
        }
        default:
            col = palette(p, t);
            break;
    }

    if (p.slopes) {
        const float ang = (p.slopeAngle + p.lightSpeed * timeSec) * 0.017453292f;
        const float lx = cosf(ang), ly = sinf(ang);
        float light = (s.nx * lx + s.ny * ly + p.slopeHeight) / (1.f + p.slopeHeight);
        light = fminf(fmaxf(light, 0.f), 1.f);
        const float shade = 1.f - p.slopeStrength * (1.f - light);
        col.x *= shade;
        col.y *= shade;
        col.z *= shade;
    }

    if (p.lines && g.valid) {
        const float v = s.iter * p.lineDensity;
        const float f = v - floorf(v);
        const float distCycles = fminf(f, 1.f - f);
        const float rate = (fabsf(g.dx) + fabsf(g.dy)) * p.lineDensity;  // cycles per pixel
        if (rate > 1e-12f) {
            const float distPx = distCycles / rate;
            const float halfW = 0.5f * p.lineWidth;
            float cover = 1.f - smoothstepf(halfW - 0.5f, halfW + 0.5f, distPx);
            cover *= fminf(fmaxf(1.5f - 2.f * halfW * rate, 0.f), 1.f);
            const float k = cover * p.lineStrength;
            col.x += (p.lineColor[0] - col.x) * k;
            col.y += (p.lineColor[1] - col.y) * k;
            col.z += (p.lineColor[2] - col.z) * k;
        }
    }

    if (p.deStrength > 0.f) {
        float edge = s.de;
        edge = fminf(fmaxf(edge, 0.f), 1.f);
        edge = powf(edge, 0.5f * p.deStrength);
        col.x *= edge;
        col.y *= edge;
        col.z *= edge;
    }
    col.x *= p.exposure;
    col.y *= p.exposure;
    col.z *= p.exposure;
    return col;
}

__device__ __forceinline__ uint32_t packSRGB8(float3 c) {
    // Linear -> sRGB, clamp, pack. HDR output will replace this.
    auto enc = [](float v) {
        v = fminf(fmaxf(v, 0.f), 1.f);
        v = v <= 0.0031308f ? 12.92f * v : 1.055f * powf(v, 1.f / 2.4f) - 0.055f;
        return (uint32_t)(v * 255.f + 0.5f);
    };
    return enc(c.x) | (enc(c.y) << 8) | (enc(c.z) << 16) | 0xFF000000u;
}
