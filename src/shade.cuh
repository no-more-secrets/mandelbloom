#pragma once
// Device-side coloring. Included by render_cuda.cu only.
#include <cstdint>
#include "field.h"

__device__ __forceinline__ float3 cosPalette(const ShadeParams& p, float t) {
    const float k = 6.2831853f;
    return make_float3(p.a[0] + p.b[0] * cosf(k * (p.c[0] * t + p.d[0])),
                       p.a[1] + p.b[1] * cosf(k * (p.c[1] * t + p.d[1])),
                       p.a[2] + p.b[2] * cosf(k * (p.c[2] * t + p.d[2])));
}

// Shade one sample to linear RGB. pixelScale is complex units per pixel,
// used to turn the distance estimate into an edge weight.
__device__ __forceinline__ float3 shadeSample(const FieldSample& s, const ShadeParams& p,
                                              float timeSec, float pixelScale) {
    // Inside the set, or not computed yet (never show a partial count: it
    // changes every slice and strobes).
    if (s.iter < 0.f || s.pad > 0.5f) return make_float3(p.inside[0], p.inside[1], p.inside[2]);

    float t;
    if (p.logScale) {
        t = log2f(fmaxf(s.iter, 1.f)) * (p.density / 64.f);
    } else {
        t = s.iter / p.density;
    }
    t += p.offset + timeSec;
    float3 col = cosPalette(p, t);

    if (p.deStrength > 0.f) {
        // Filaments thinner than a pixel still get drawn dark.
        float edge = s.de / pixelScale;
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
