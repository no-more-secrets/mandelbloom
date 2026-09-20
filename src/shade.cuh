#pragma once
// Device-side coloring. Included by render_cuda.cu only.
// Shading is split in two. prepareSample() turns a field sample into a
// ShadeInput: everything that does not depend on time (palette position,
// edge factor, line coverage, normal). colourInput() turns a ShadeInput
// into a colour for a given time. The live composite runs both per frame;
// the settled path caches ShadeInputs once and runs only colourInput().
#include <cstdint>
#include <cuda_fp16.h>
#include "field.h"

#define PALETTE_LUT_SIZE 1024

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }

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

// Direct evaluation; used to fill the lookup table.
__device__ __forceinline__ float3 paletteDirect(const ShadeParams& p, float t) {
    return p.paletteType == 1 ? cosPalette(p, t) : gradientPalette(p, t);
}

// One palette cycle tabulated: linear interpolation with wrap.
__device__ __forceinline__ float3 paletteLut(const float4* __restrict__ lut, float t) {
    const float u = fracf(t) * PALETTE_LUT_SIZE;
    const int i0 = (int)u;
    const int i1 = (i0 + 1) & (PALETTE_LUT_SIZE - 1);
    const float f = u - i0;
    const float4 a = lut[i0 & (PALETTE_LUT_SIZE - 1)], b = lut[i1];
    return make_float3(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.z + (b.z - a.z) * f);
}

__device__ __forceinline__ float hash01(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (float)(x & 0xFFFFFFu) / 16777216.f;
}

// Subsample offset in field pixels (each within +-0.5) for field pixel
// (fx, fy) whose subsample index inside its display pixel is (si, sj).
// origin keeps stochastic offsets attached to their samples across field
// shifts (a sample's key is fx + ox, fy + oy).
__device__ __forceinline__ float2 sampleJitter(int pattern, int ss, int fx, int fy, int si,
                                               int sj, int ox, int oy) {
    if (pattern == AA_ROTATED && ss == 2) {
        // Rotated grid: four distinct x and y coverage levels per pixel.
        const float jx[4] = {0.25f, 0.25f, -0.25f, -0.25f};
        const float jy[4] = {-0.25f, 0.25f, -0.25f, 0.25f};
        const int k = sj * 2 + si;
        return make_float2(jx[k], jy[k]);
    }
    if (pattern == AA_STOCHASTIC || (pattern == AA_ROTATED && ss != 2)) {
        const uint32_t key = (uint32_t)(fx + ox) * 73856093u ^ (uint32_t)(fy + oy) * 19349663u;
        return make_float2(hash01(key) - 0.5f, hash01(key ^ 0x9E3779B9u) - 0.5f);
    }
    return make_float2(0.f, 0.f);
}

// Reconstruction filter weight for a sample at distance d (display pixels)
// from the pixel centre; r is the filter radius.
__device__ __forceinline__ float filterWeight(int filter, float d, float r) {
    const float t = d / r;
    if (t >= 1.f) return 0.f;
    switch (filter) {
        case FILTER_TENT:
            return 1.f - t;
        case FILTER_GAUSSIAN: {
            // sigma = r/2, truncated at r and shifted to zero there
            const float g = expf(-2.f * t * t), g1 = expf(-2.f);
            return (g - g1) / (1.f - g1);
        }
        case FILTER_BLACKMAN: {
            const float a = 3.14159265f * (t + 1.f);  // window over [-r, r]
            return 0.35875f - 0.48829f * cosf(a) + 0.14128f * cosf(2.f * a) -
                   0.01168f * cosf(3.f * a);
        }
        default:
            return 1.f;
    }
}

// Screen-space rate of change of the iteration count, per display pixel,
// used to draw lines at constant width. Zero when unknown.
struct IterGradient {
    float dx = 0.f, dy = 0.f;
    bool valid = false;
};

// Time-independent shading data for one sample. 16 bytes.
struct ShadeInput {
    float t;          // palette position in cycles, before the time term
    uint16_t edge;    // half: distance-estimate edge factor, already powered
    uint16_t line;    // half: line coverage 0..1, before lineStrength
    uint16_t nx, ny;  // half: exterior normal
    uint16_t flags;   // 0 inside or no data, 1 exterior
    uint16_t pad;
};

__device__ __forceinline__ ShadeInput prepareSample(const FieldSample& s, const ShadeParams& p,
                                                    const IterGradient& g) {
    ShadeInput in{};
    if (s.iter < 0.f || s.gen == 0) return in;  // flags 0
    in.flags = 1;
    const float de = h2f(s.de);

    float it = s.iter;
    if (p.anchor) it = fmaxf(it - p.iterBase, 0.f);
    float t;
    switch (p.transfer) {
        case TRANSFER_LINEAR:
            t = it / p.density;
            break;
        case TRANSFER_SQRT:
            t = sqrtf(fmaxf(it, 0.f)) * (8.f / p.density);
            break;
        default:
            t = log2f(fmaxf(it, 1.f)) * (p.density / 64.f);
            break;
    }
    switch (p.mode) {
        case SHADE_ANGLE:
            t = h2f(s.angle) * 0.15915494f * p.special;  // turns * cycles per turn
            break;
        case SHADE_DISTANCE:
            t = log10f(fmaxf(de, 1e-6f)) / fmaxf(p.special, 0.01f);
            break;
        default:
            break;
    }
    in.t = t + p.offset;

    float edge = 1.f;
    if (p.deStrength > 0.f) {
        edge = fminf(fmaxf(de, 0.f), 1.f);
        edge = powf(edge, 0.5f * p.deStrength);
    }
    in.edge = f2h(edge);

    float cover = 0.f;
    if (p.lines && g.valid) {
        const float v = s.iter * p.lineDensity;
        const float f = v - floorf(v);
        const float distCycles = fminf(f, 1.f - f);
        const float rate = (fabsf(g.dx) + fabsf(g.dy)) * p.lineDensity;  // cycles per pixel
        if (rate > 1e-12f) {
            const float distPx = distCycles / rate;
            const float halfW = 0.5f * p.lineWidth;
            cover = 1.f - smoothstepf(halfW - 0.5f, halfW + 0.5f, distPx);
            cover *= fminf(fmaxf(1.5f - 2.f * halfW * rate, 0.f), 1.f);
        }
    }
    in.line = f2h(cover);
    in.nx = s.nx;
    in.ny = s.ny;
    return in;
}

// Colour for a ShadeInput at a given time. Only time-dependent and cheap
// parameters are applied here.
__device__ __forceinline__ float3 colourInput(const ShadeInput& in, const ShadeParams& p,
                                              float timeSec, const float4* __restrict__ lut) {
    if (in.flags == 0) return make_float3(p.inside[0], p.inside[1], p.inside[2]);
    const float tt = in.t + p.cycleSpeed * timeSec;
    float3 col = paletteLut(lut, tt);

    switch (p.mode) {
        case SHADE_LOGSTEPS: {
            const float f = fracf(tt * p.special + p.wavePhase + p.waveSpeed * timeSec);
            const float k = 0.25f + 0.75f * logf(1.f + f * 1.7182818f);
            col.x *= k; col.y *= k; col.z *= k;
            break;
        }
        case SHADE_WAVE: {
            const float w = 0.5f + 0.5f * sinf(6.2831853f * (tt * p.special + p.wavePhase + p.waveSpeed * timeSec));
            col.x *= w; col.y *= w; col.z *= w;
            break;
        }
        case SHADE_PANELS: {
            const float f = fracf(tt * p.special + p.wavePhase + p.waveSpeed * timeSec);
            const float gap = 0.15f;
            const float k = smoothstepf(0.f, gap, f) * smoothstepf(1.f, 1.f - gap, f);
            const float m = 0.1f + 0.9f * k;
            col.x *= m; col.y *= m; col.z *= m;
            break;
        }
        default:
            break;
    }

    if (p.slopes) {
        const float ang = (p.slopeAngle + p.lightSpeed * timeSec) * 0.017453292f;
        const float lx = cosf(ang), ly = sinf(ang);
        float light = (h2f(in.nx) * lx + h2f(in.ny) * ly + p.slopeHeight) / (1.f + p.slopeHeight);
        light = fminf(fmaxf(light, 0.f), 1.f);
        const float shade = 1.f - p.slopeStrength * (1.f - light);
        col.x *= shade; col.y *= shade; col.z *= shade;
    }

    const float k = h2f(in.line) * p.lineStrength;
    if (k > 0.f) {
        col.x += (p.lineColor[0] - col.x) * k;
        col.y += (p.lineColor[1] - col.y) * k;
        col.z += (p.lineColor[2] - col.z) * k;
    }

    const float edge = h2f(in.edge) * p.exposure;
    col.x *= edge; col.y *= edge; col.z *= edge;
    return col;
}

__device__ __forceinline__ float3 shadeSample(const FieldSample& s, const ShadeParams& p,
                                              float timeSec, float /*pixelScale*/,
                                              const IterGradient& g,
                                              const float4* __restrict__ lut) {
    return colourInput(prepareSample(s, p, g), p, timeSec, lut);
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
