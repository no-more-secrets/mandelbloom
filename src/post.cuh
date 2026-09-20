#pragma once
// Post-processing on the linear HDR image. Included by render_cuda.cu.
//
// Pipeline: shading writes linear colour into hdr (RGBA16F, display size).
// postBloomDown extracts bright pixels at 1/4 resolution, blurH/blurV
// blur that (twice: a tight level and a wide level at 1/8), and postFinal
// composes bloom, aberration, vignette, grading, grain, tone mapping and
// the scRGB white scale into the output buffer.
#include <cstdint>
#include "field.h"
#include "shade.cuh"

namespace post {

__device__ __forceinline__ float3 loadHdr(const uint16_t* __restrict__ hdr, int w, int x, int y) {
    const ushort4 v = *reinterpret_cast<const ushort4*>(hdr + ((size_t)y * w + x) * 4);
    return make_float3(h2f(v.x), h2f(v.y), h2f(v.z));
}

__device__ __forceinline__ float3 sampleHdr(const uint16_t* __restrict__ hdr, int w, int h,
                                            float fx, float fy) {
    fx = fminf(fmaxf(fx, 0.f), (float)(w - 1));
    fy = fminf(fmaxf(fy, 0.f), (float)(h - 1));
    const int x0 = (int)fx, y0 = (int)fy;
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float tx = fx - x0, ty = fy - y0;
    const float3 a = loadHdr(hdr, w, x0, y0), b = loadHdr(hdr, w, x1, y0);
    const float3 c = loadHdr(hdr, w, x0, y1), d = loadHdr(hdr, w, x1, y1);
    return make_float3(
        (a.x * (1 - tx) + b.x * tx) * (1 - ty) + (c.x * (1 - tx) + d.x * tx) * ty,
        (a.y * (1 - tx) + b.y * tx) * (1 - ty) + (c.y * (1 - tx) + d.y * tx) * ty,
        (a.z * (1 - tx) + b.z * tx) * (1 - ty) + (c.z * (1 - tx) + d.z * tx) * ty);
}

__device__ __forceinline__ float3 sampleF4(const float4* __restrict__ img, int w, int h, float fx,
                                           float fy) {
    fx = fminf(fmaxf(fx, 0.f), (float)(w - 1));
    fy = fminf(fmaxf(fy, 0.f), (float)(h - 1));
    const int x0 = (int)fx, y0 = (int)fy;
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float tx = fx - x0, ty = fy - y0;
    const float4 a = img[(size_t)y0 * w + x0], b = img[(size_t)y0 * w + x1];
    const float4 c = img[(size_t)y1 * w + x0], d = img[(size_t)y1 * w + x1];
    return make_float3(
        (a.x * (1 - tx) + b.x * tx) * (1 - ty) + (c.x * (1 - tx) + d.x * tx) * ty,
        (a.y * (1 - tx) + b.y * tx) * (1 - ty) + (c.y * (1 - tx) + d.y * tx) * ty,
        (a.z * (1 - tx) + b.z * tx) * (1 - ty) + (c.z * (1 - tx) + d.z * tx) * ty);
}

__device__ __forceinline__ float luma(float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

__device__ __forceinline__ float hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (float)(x & 0xFFFFFFu) / 16777216.f;
}

// ACES fitted (Krzysztof Narkowicz), input linear, output 0..1.
__device__ __forceinline__ float acesFit(float x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return fminf(fmaxf((x * (a * x + b)) / (x * (c * x + d) + e), 0.f), 1.f);
}

}  // namespace post

// Bright pass + 4x4 box downsample: hdr (w x h) -> bloom (bw x bh).
__global__ void postBloomDown(const uint16_t* __restrict__ hdr, int w, int h, float4* __restrict__ out,
                              int bw, int bh, float threshold, float knee) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= bw || y >= bh) return;
    float3 acc = make_float3(0.f, 0.f, 0.f);
    int n = 0;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            const int sx = x * 4 + i, sy = y * 4 + j;
            if (sx >= w || sy >= h) continue;
            const float3 c = post::loadHdr(hdr, w, sx, sy);
            const float l = post::luma(c);
            // Soft knee around the threshold.
            const float soft = fminf(fmaxf(l - threshold + knee, 0.f), 2.f * knee);
            float contrib = fmaxf(l - threshold, soft * soft / (4.f * knee + 1e-5f));
            contrib = fmaxf(contrib, 0.f) / fmaxf(l, 1e-5f);
            acc.x += c.x * contrib;
            acc.y += c.y * contrib;
            acc.z += c.z * contrib;
            ++n;
        }
    }
    const float inv = n > 0 ? 1.f / n : 0.f;
    out[(size_t)y * bw + x] = make_float4(acc.x * inv, acc.y * inv, acc.z * inv, 0.f);
}

// 2x box downsample of a float4 image.
__global__ void postDown2(const float4* __restrict__ in, int iw, int ih, float4* __restrict__ out,
                          int ow, int oh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ow || y >= oh) return;
    float4 acc = make_float4(0, 0, 0, 0);
    int n = 0;
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            const int sx = min(x * 2 + i, iw - 1), sy = min(y * 2 + j, ih - 1);
            const float4 v = in[(size_t)sy * iw + sx];
            acc.x += v.x; acc.y += v.y; acc.z += v.z;
            ++n;
        }
    const float inv = 1.f / n;
    out[(size_t)y * ow + x] = make_float4(acc.x * inv, acc.y * inv, acc.z * inv, 0.f);
}

// Separable Gaussian blur, radius in pixels of the buffer (<= 24).
__global__ void postBlur(const float4* __restrict__ in, float4* __restrict__ out, int w, int h,
                         int radius, int dx, int dy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    radius = min(max(radius, 1), 24);
    const float sigma = radius / 2.5f;
    float3 acc = make_float3(0, 0, 0);
    float wsum = 0.f;
    for (int k = -radius; k <= radius; ++k) {
        const int sx = min(max(x + k * dx, 0), w - 1), sy = min(max(y + k * dy, 0), h - 1);
        const float wgt = expf(-0.5f * (float)(k * k) / (sigma * sigma));
        const float4 v = in[(size_t)sy * w + sx];
        acc.x += v.x * wgt; acc.y += v.y * wgt; acc.z += v.z * wgt;
        wsum += wgt;
    }
    const float inv = 1.f / wsum;
    out[(size_t)y * w + x] = make_float4(acc.x * inv, acc.y * inv, acc.z * inv, 0.f);
}

// Final composition into the output buffer.
__global__ void postFinal(const uint16_t* __restrict__ hdr, int w, int h,
                          const float4* __restrict__ bloomA, int aw, int ah,
                          const float4* __restrict__ bloomB, int bw, int bh,
                          uint16_t* __restrict__ out, int pitchPx, PostParams p, float whiteScale,
                          float headroom, uint32_t frame) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // Normalised radial coordinate for vignette and aberration.
    const float cx = 0.5f * w, cy = 0.5f * h;
    const float rx = (x - cx) / cx, ry = (y - cy) / cy;
    const float r2 = rx * rx + ry * ry;

    float3 c;
    if (p.aberration > 0.f) {
        // Red and blue sampled slightly in/out along the radius.
        const float shift = p.aberration * r2;
        const float dxp = rx * shift, dyp = ry * shift;
        const float3 cr = post::sampleHdr(hdr, w, h, x - dxp, y - dyp);
        const float3 cg = post::loadHdr(hdr, w, x, y);
        const float3 cb = post::sampleHdr(hdr, w, h, x + dxp, y + dyp);
        c = make_float3(cr.x, cg.y, cb.z);
    } else {
        c = post::loadHdr(hdr, w, x, y);
    }

    if (p.sharpen > 0.f) {
        // Unsharp mask against the 3x3 neighbourhood mean.
        float3 m = make_float3(0, 0, 0);
        for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i) {
                const float3 s = post::loadHdr(hdr, w, min(max(x + i, 0), w - 1),
                                               min(max(y + j, 0), h - 1));
                m.x += s.x; m.y += s.y; m.z += s.z;
            }
        m.x /= 9.f; m.y /= 9.f; m.z /= 9.f;
        c.x = fmaxf(c.x + (c.x - m.x) * p.sharpen, 0.f);
        c.y = fmaxf(c.y + (c.y - m.y) * p.sharpen, 0.f);
        c.z = fmaxf(c.z + (c.z - m.z) * p.sharpen, 0.f);
    }

    if (p.bloomIntensity > 0.f) {
        const float3 a = post::sampleF4(bloomA, aw, ah, (x + 0.5f) / 4.f - 0.5f, (y + 0.5f) / 4.f - 0.5f);
        const float3 b = post::sampleF4(bloomB, bw, bh, (x + 0.5f) / 8.f - 0.5f, (y + 0.5f) / 8.f - 0.5f);
        c.x += p.bloomIntensity * (a.x + p.bloomWide * b.x);
        c.y += p.bloomIntensity * (a.y + p.bloomWide * b.y);
        c.z += p.bloomIntensity * (a.z + p.bloomWide * b.z);
    }

    // Grading.
    c.x *= p.gain; c.y *= p.gain; c.z *= p.gain;
    if (p.saturation != 1.f) {
        const float l = post::luma(c);
        c.x = l + (c.x - l) * p.saturation;
        c.y = l + (c.y - l) * p.saturation;
        c.z = l + (c.z - l) * p.saturation;
    }
    if (p.contrast != 1.f) {
        // Pivot at mid grey 0.18.
        c.x = fmaxf(0.18f * powf(fmaxf(c.x, 0.f) / 0.18f, p.contrast), 0.f);
        c.y = fmaxf(0.18f * powf(fmaxf(c.y, 0.f) / 0.18f, p.contrast), 0.f);
        c.z = fmaxf(0.18f * powf(fmaxf(c.z, 0.f) / 0.18f, p.contrast), 0.f);
    }
    if (p.vignette > 0.f) {
        const float v = 1.f - p.vignette * smoothstepf(1.f - p.vignetteSoft, 1.4f, sqrtf(r2));
        c.x *= v; c.y *= v; c.z *= v;
    }
    if (p.grain > 0.f) {
        const uint32_t seed = (uint32_t)(y * w + x) * 2654435761u ^ (frame * 40503u);
        const float n = (post::hash(seed) - 0.5f) * p.grain;
        c.x = fmaxf(c.x + n * (0.2f + c.x), 0.f);
        c.y = fmaxf(c.y + n * (0.2f + c.y), 0.f);
        c.z = fmaxf(c.z + n * (0.2f + c.z), 0.f);
    }

    // Tone mapping into [0, headroom] (headroom = 1 on SDR displays), then
    // to scRGB by the SDR white scale.
    const float peak = fmaxf(headroom, 1.f);
    c.x *= p.hdrBoost; c.y *= p.hdrBoost; c.z *= p.hdrBoost;
    switch (p.tonemap) {
        case 1: {  // Reinhard extended, white at the peak
            const float wp = peak * peak;
            c.x = c.x * (1.f + c.x / wp) / (1.f + c.x);
            c.y = c.y * (1.f + c.y / wp) / (1.f + c.y);
            c.z = c.z * (1.f + c.z / wp) / (1.f + c.z);
            break;
        }
        case 2: {  // ACES fitted, scaled to the peak
            c.x = peak * post::acesFit(c.x / peak);
            c.y = peak * post::acesFit(c.y / peak);
            c.z = peak * post::acesFit(c.z / peak);
            break;
        }
        default:
            c.x = fminf(c.x, peak); c.y = fminf(c.y, peak); c.z = fminf(c.z, peak);
            break;
    }
    ushort4 v;
    v.x = f2h(fmaxf(c.x, 0.f) * whiteScale);
    v.y = f2h(fmaxf(c.y, 0.f) * whiteScale);
    v.z = f2h(fmaxf(c.z, 0.f) * whiteScale);
    v.w = f2h(1.f);
    *reinterpret_cast<ushort4*>(out + ((size_t)y * pitchPx + x) * 4) = v;
}
