#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <utility>
#include <vector>
#include "render_cuda.h"
#include "bla.h"
#include "shade.cuh"
#include "post.cuh"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                         __LINE__, cudaGetErrorString(err__));                    \
            return false;                                                         \
        }                                                                         \
    } while (0)

// Per-pixel iteration state carried between slices.
struct PixelState {
    double dzr, dzi;  // dz = z - Z_m
    double dr, di;    // dz/dc * scale, for the distance estimate
    int m;            // reference index
    int n;            // pixel iteration
    int status;       // 0 active, 1 escaped, 2 inside
    int flags;        // bit 0: ran past the end of an escaped reference (unreliable)
};

namespace {

// GPU time per slice launch. Short enough that the composite on the other
// stream and any pending view change wait at most this long.
constexpr unsigned long long kSliceBudgetNs = 20ull * 1000000ull;

__device__ __forceinline__ FieldSample emptySample() { return FieldSample{}; }  // gen 0

// New pass: reset iteration state only. Field samples from earlier
// generations stay until overwritten.
__global__ void resetState(PixelState* __restrict__ st, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    st[i] = PixelState{};
}

__global__ void clearField(FieldSample* __restrict__ field, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    field[i] = emptySample();
}

// Count pixels still to compute at a stride level (same grid rule as the
// slice kernel), for the progress model.
__global__ void countLevelKernel(const PixelState* __restrict__ state, int fw, int fh, int mx,
                                 int my, int w, int h, int stride, int statusOffsetInts,
                                 int* __restrict__ out) {
    int x, y;
    bool inBounds;
    if (stride == 1) {
        x = mx + blockIdx.x * blockDim.x + threadIdx.x;
        y = my + blockIdx.y * blockDim.y + threadIdx.y;
        inBounds = x < mx + w && y < my + h;
    } else {
        x = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
        y = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
        inBounds = x < fw && y < fh;
    }
    // The two state layouts keep status at different offsets.
    const bool active =
        inBounds &&
        reinterpret_cast<const int*>(state + ((size_t)y * fw + x))[statusOffsetInts] == 0;
    const unsigned mask = __ballot_sync(0xffffffffu, active);
    if ((threadIdx.x & 31) == 0 && mask) atomicAdd(out, __popc(mask));
}

// Reset only pixels that have not finished (their dz referred to an old
// reference orbit).
__global__ void resetPendingState(PixelState* __restrict__ st, int count, int statusOffsetInts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int* w = reinterpret_cast<int*>(st + i);
    // flags is the last int in both layouts (offset 44).
    if (w[statusOffsetInts] == 0 || (w[11] & 1)) st[i] = PixelState{};
}

// Find one pixel flagged unreliable (lowest index). out = -1 if none.
__global__ void findUnreliableKernel(const PixelState* __restrict__ st, int count,
                                     int* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int* w = reinterpret_cast<const int*>(st + i);
    if (w[11] & 1) atomicMin(out, i);
}

// dst(x, y) = src(x + dx, y + dy), pending where the source is off-image.
__global__ void shiftKernel(const PixelState* __restrict__ srcS,
                            const FieldSample* __restrict__ srcF, PixelState* __restrict__ dstS,
                            FieldSample* __restrict__ dstF, int w, int h, int dx, int dy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int sx = x + dx, sy = y + dy;
    const size_t di = (size_t)y * w + x;
    if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
        const size_t si = (size_t)sy * w + sx;
        dstS[di] = srcS[si];
        dstF[di] = srcF[si];
    } else {
        dstS[di] = PixelState{};
        dstF[di] = emptySample();
    }
}

// Returns the escape iteration (negative when the pixel is inside).
__device__ __forceinline__ float writeSample(FieldSample* __restrict__ field, size_t idx,
                                             const PixelState& st, double zr, double zi,
                                             double scale, bool escaped, int gen) {
    FieldSample s{};
    if (!escaped) {
        s.iter = -1.f;
    } else {
        const double mag2 = zr * zr + zi * zi;
        const double logMag = 0.5 * log(mag2);
        s.iter = (float)(st.n + 1.0 - log2(logMag / 0.6931471805599453));
        const double dmag2 = st.dr * st.dr + st.di * st.di;
        const double dmag = sqrt(dmag2);
        // Derivative was scaled by pixel size, so this DE is in pixels.
        s.de = f2h(dmag > 0.0 ? (float)(sqrt(mag2) * logMag / dmag) : 0.f);
        s.angle = f2h((float)atan2(zi, zr));
        // Milnor normal u = z / dz, normalised.
        if (dmag2 > 0.0) {
            const double ur = (zr * st.dr + zi * st.di) / dmag2;
            const double ui = (zi * st.dr - zr * st.di) / dmag2;
            const double um = sqrt(ur * ur + ui * ui);
            if (um > 0.0) {
                s.nx = f2h((float)(ur / um));
                s.ny = f2h((float)(ui / um));
            }
        }
    }
    s.gen = (uint16_t)gen;
    field[idx] = s;
    return s.iter;
}

__device__ __forceinline__ unsigned long long globalTimerNs() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Stamps the GPU clock right before the slice so the deadline is relative
// to when the GPU actually started it, not when the host queued it.
__global__ void stampTimer(unsigned long long* __restrict__ out) { *out = globalTimerNs(); }

}  // namespace
#include "iterate_float.cuh"
namespace {

// Find the longest BLA node starting at reference index m that is valid
// for |dz|^2 = dzmag2. Returns nullptr if none.
__device__ __forceinline__ const BlaNode* findBla(const DeviceBla& bla, int m, double dzmag2,
                                                  int refLast) {
    const int j0 = m - 1;
    if (j0 < 0 || j0 >= bla.steps) return nullptr;
    int k = j0 == 0 ? bla.levels - 1 : min(bla.levels - 1, __ffs(j0) - 1);
    for (; k >= 0; --k) {
        const BlaNode* n = bla.nodes + bla.levelOffset[k] + (j0 >> k);
        if (dzmag2 < n->r2 && m + n->l <= refLast) return n;
    }
    return nullptr;
}

// Perturbation iteration in double precision with Zhuoran's rebasing and
// BLA skipping, advanced by at most sliceIters iterations per launch and
// never past the GPU-clock deadline.
//   dz_{n+1} = 2 Z_m dz_n + dz_n^2 + dc
//   rebase when |Z_m + dz| < |dz|: dz = Z_m + dz, m = 0
// Field layout: fw x fh with the view (w x h) at offset (mx, my). Pixel
// (x, y) in field coordinates sits at view offset (x - mx, y - my).
__global__ void iterateSlice(PixelState* __restrict__ state, FieldSample* __restrict__ field,
                             int fw, int fh, int mx, int my, int w, int h, int stride,
                             double scale, double refOffX, double refOffY, int maxIter,
                             int sliceIters, const unsigned long long* __restrict__ startNs,
                             unsigned long long budgetNs, DeviceReference ref, DeviceBla bla,
                             int gen, int ss, int aaPattern, int jox, int joy,
                             int* __restrict__ activeCount, int* __restrict__ minIter) {
    const unsigned long long deadlineNs = *startNs + budgetNs;
    float myMin = 3.0e38f;  // smallest iteration count this thread escaped at
    int x, y;
    bool inBounds;
    if (stride == 1) {
        // Full resolution only inside the view; the margin stays coarse.
        x = mx + blockIdx.x * blockDim.x + threadIdx.x;
        y = my + blockIdx.y * blockDim.y + threadIdx.y;
        inBounds = x < mx + w && y < my + h;
    } else {
        x = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
        y = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
        inBounds = x < fw && y < fh;
    }
    const size_t idx = inBounds ? (size_t)y * fw + x : 0;
    PixelState st{};
    if (inBounds) st = state[idx];
    bool active = inBounds && st.status == 0;

    if (active) {
        const float2 jit = sampleJitter(aaPattern, ss, x, y, ((x - mx) % ss + ss) % ss,
                                        ((y - my) % ss + ss) % ss, jox, joy);
        const double dcr = refOffX + ((double)(x - mx) + jit.x - 0.5 * w) * scale;
        const double dci = refOffY - ((double)(y - my) + jit.y - 0.5 * h) * scale;
        double dzr = st.dzr, dzi = st.dzi, dr = st.dr, di = st.di;
        int m = st.m, n = st.n;
        double zr = 0.0, zi = 0.0;
        const double bailout = 65536.0;
        const int refLast = ref.length - 1;
        const bool refShort = refLast < maxIter;  // reference escaped early
        bool unreliable = false;
        const int stop = min(maxIter, n + sliceIters);
        bool escaped = false;
        double dzmag2 = dzr * dzr + dzi * dzi;
        int budgetCheck = 0;

        while (n < stop) {
            if ((++budgetCheck & 63) == 0 && globalTimerNs() > deadlineNs) break;
            if (bla.enabled) {
                const BlaNode* nd = findBla(bla, m, dzmag2, refLast);
                if (nd) {
                    // dz' = A dz + B dc ; d' = A d + B scale
                    const double ndzr = nd->ar * dzr - nd->ai * dzi + nd->br * dcr - nd->bi * dci;
                    const double ndzi = nd->ar * dzi + nd->ai * dzr + nd->br * dci + nd->bi * dcr;
                    const double ndr = nd->ar * dr - nd->ai * di + nd->br * scale;
                    const double ndi = nd->ar * di + nd->ai * dr + nd->bi * scale;
                    dzr = ndzr;
                    dzi = ndzi;
                    dr = ndr;
                    di = ndi;
                    m += nd->l;
                    n += nd->l;
                    zr = ref.zr[m] + dzr;
                    zi = ref.zi[m] + dzi;
                    const double zmag2 = zr * zr + zi * zi;
                    if (zmag2 > bailout) {
                        escaped = true;
                        break;
                    }
                    dzmag2 = dzr * dzr + dzi * dzi;
                    if (zmag2 < dzmag2 || m >= refLast) {
                        if (m >= refLast && refShort) unreliable = true;
                        dzr = zr;
                        dzi = zi;
                        dzmag2 = zmag2;
                        m = 0;
                    }
                    continue;
                }
            }
            const double Zr = ref.zr[m];
            const double Zi = ref.zi[m];
            const double fzr = Zr + dzr, fzi = Zi + dzi;
            const double ndr = 2.0 * (fzr * dr - fzi * di) + scale;
            const double ndi = 2.0 * (fzr * di + fzi * dr);
            dr = ndr;
            di = ndi;
            const double ar = 2.0 * Zr + dzr, ai = 2.0 * Zi + dzi;
            const double ndzr = ar * dzr - ai * dzi + dcr;
            const double ndzi = ar * dzi + ai * dzr + dci;
            dzr = ndzr;
            dzi = ndzi;
            ++m;
            ++n;
            zr = ref.zr[m] + dzr;
            zi = ref.zi[m] + dzi;
            const double zmag2 = zr * zr + zi * zi;
            if (zmag2 > bailout) {
                escaped = true;
                break;
            }
            dzmag2 = dzr * dzr + dzi * dzi;
            if (zmag2 < dzmag2 || m >= refLast) {
                if (m >= refLast && refShort) unreliable = true;
                dzr = zr;
                dzi = zi;
                dzmag2 = zmag2;
                m = 0;
            }
        }
        if (unreliable) st.flags |= 1;
        st.dzr = dzr;
        st.dzi = dzi;
        st.dr = dr;
        st.di = di;
        st.m = m;
        st.n = n;
        if (escaped) st.status = 1;
        else if (n >= maxIter) st.status = 2;
        const bool pending = st.status == 0;
        // In-progress pixels keep whatever older sample sits there.
        if (!pending) {
            const float it = writeSample(field, idx, st, zr, zi, scale, escaped, gen);
            if (it >= 0.f) myMin = fminf(myMin, it);
        }
        state[idx] = st;
        active = pending;
    }

    // Warp-aggregated count of pixels still active.
    const unsigned mask = __ballot_sync(0xffffffffu, active);
    if ((threadIdx.x & 31) == 0 && mask) atomicAdd(activeCount, __popc(mask));
    // Warp-reduced minimum escape iteration (non-negative floats order as ints).
    for (int o = 16; o > 0; o >>= 1) myMin = fminf(myMin, __shfl_xor_sync(0xffffffffu, myMin, o));
    if ((threadIdx.x & 31) == 0 && myMin < 3.0e38f) atomicMin(minIter, __float_as_int(myMin));
}

// Best sample of generation g at field pixel (x, y): the pixel itself, or
// the nearest coarse-level anchor (stride 2, 4, 8) if the pixel itself was
// not produced by that generation. level receives 1, 2, 4, 8 (0 = none).
__device__ __forceinline__ FieldSample fetchGen(const FieldSample* __restrict__ field, int fw,
                                                int x, int y, int g, int& level) {
    FieldSample s = field[(size_t)y * fw + x];
    if ((int)s.gen == g) {
        level = 1;
        return s;
    }
    for (int k = 2; k <= 8; k *= 2) {
        const int bx = x - (x % k), by = y - (y % k);
        const FieldSample a = field[(size_t)by * fw + bx];
        if ((int)a.gen == g) {
            level = k;
            return a;
        }
    }
    level = 0;
    return s;
}

// Shade one field sample (with same-generation neighbour gradient for lines).
__device__ __forceinline__ IterGradient gradientAt(const FieldSample* __restrict__ field, int fw,
                                                   int fh, int xi, int yi, const FieldSample& s,
                                                   const ShadeParams& p, const GenMap& gm) {
    IterGradient g;
    if (p.lines && xi + 1 < fw && yi + 1 < fh && s.iter >= 0.f) {
        const FieldSample sx = field[(size_t)yi * fw + xi + 1];
        const FieldSample sy = field[(size_t)(yi + 1) * fw + xi];
        if (sx.gen == s.gen && sy.gen == s.gen && sx.iter >= 0.f && sy.iter >= 0.f) {
            g.dx = (sx.iter - s.iter) * (float)gm.ratio;
            g.dy = (sy.iter - s.iter) * (float)gm.ratio;
            g.valid = true;
        }
    }
    return g;
}

__device__ __forceinline__ float3 shadeField(const FieldSample* __restrict__ field, int fw, int fh,
                                             int xi, int yi, const FieldSample& s,
                                             const ShadeParams& p, float timeSec,
                                             const GenMap& gm, const float4* __restrict__ lut) {
    return shadeSample(s, p, timeSec, gm.pixelScale, gradientAt(field, fw, fh, xi, yi, s, p, gm),
                       lut);
}

// One subsample: search the generations for the best data at this point.
// Returns false if no generation has anything here; otherwise the sample,
// its field position and the generation it came from.
__device__ __forceinline__ bool resolveBest(const FieldSample* __restrict__ field, int fw, int fh,
                                            int mx, int my, double px, double py,
                                            const CompositeMap& map, FieldSample& out, int& outX,
                                            int& outY, int& outGen) {
    float bestDetail = 0.f;
    FieldSample best{};
    int bestX = 0, bestY = 0, bestGen = -1;
    for (int k = 0; k < map.genCount; ++k) {
        const GenMap& gm = map.gens[k];
        const double u = gm.ox + px * gm.ratio + mx, v = gm.oy + py * gm.ratio + my;
        const int xi = (int)floor(u + 0.5), yi = (int)floor(v + 0.5);
        if (xi < 0 || xi >= fw || yi < 0 || yi >= fh) continue;
        int level = 0;
        const FieldSample s = fetchGen(field, fw, xi, yi, gm.gen, level);
        if (level == 0) continue;
        const float detail = (float)gm.ratio / (float)level;  // field px per display px
        if (detail > bestDetail) {
            bestDetail = detail;
            best = s;
            bestX = xi;
            bestY = yi;
            bestGen = k;
        }
        if (level == 1 && k + 1 < map.genCount && map.gens[k + 1].ratio <= gm.ratio) break;
    }
    if (bestGen < 0) {
        // Nothing covers this point: extend the nearest field edge of the
        // first generation with data there.
        for (int k = 0; k < map.genCount; ++k) {
            const GenMap& gm = map.gens[k];
            const double u = gm.ox + px * gm.ratio + mx, v = gm.oy + py * gm.ratio + my;
            const int xi = min(max((int)floor(u + 0.5), 0), fw - 1);
            const int yi = min(max((int)floor(v + 0.5), 0), fh - 1);
            int level = 0;
            const FieldSample sm = fetchGen(field, fw, xi, yi, gm.gen, level);
            if (level == 0) continue;
            out = sm;
            outX = xi;
            outY = yi;
            outGen = k;
            return true;
        }
        return false;
    }
    out = best;
    outX = bestX;
    outY = bestY;
    outGen = bestGen;
    return true;
}

__device__ __forceinline__ bool sampleBest(const FieldSample* __restrict__ field, int fw, int fh,
                                           int mx, int my, double px, double py,
                                           const CompositeMap& map, const ShadeParams& p,
                                           float timeSec, const float4* __restrict__ lut,
                                           float3& col) {
    FieldSample s;
    int xi, yi, gi;
    if (!resolveBest(field, fw, fh, mx, my, px, py, map, s, xi, yi, gi)) return false;
    col = shadeField(field, fw, fh, xi, yi, s, p, timeSec, map.gens[gi], lut);
    return true;
}

// Write a linear colour as RGBA16F into the shading result buffer.
__device__ __forceinline__ void storeOut(uint16_t* __restrict__ out, int pitchPx, int x, int y,
                                         float3 c, float scale) {
    ushort4 v;
    v.x = f2h(c.x * scale);
    v.y = f2h(c.y * scale);
    v.z = f2h(c.z * scale);
    v.w = f2h(1.f);
    *reinterpret_cast<ushort4*>(out + ((size_t)y * pitchPx + x) * 4) = v;
}

// Display composite: ss x ss subsamples per display pixel, each taken from
// the generation with the most detail at that point.
__global__ void compositeKernel(const FieldSample* __restrict__ field, int fw, int fh, int mx,
                                int my, uint16_t* __restrict__ out, int pitchPx, float outScale,
                                int w, int h, ShadeParams p, float timeSec, CompositeMap map,
                                const float4* __restrict__ lut) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t idx = (size_t)y * w + x;
    const double px = (double)x - 0.5 * w, py = (double)y - 0.5 * h;
    const int ss = map.ss < 1 ? 1 : map.ss;
    float3 acc = make_float3(0.f, 0.f, 0.f);
    int have = 0;
    for (int j = 0; j < ss; ++j) {
        for (int i = 0; i < ss; ++i) {
            const double sx = px + (i + 0.5) / ss - 0.5, sy = py + (j + 0.5) / ss - 0.5;
            float3 c;
            if (sampleBest(field, fw, fh, mx, my, sx, sy, map, p, timeSec, lut, c)) {
                acc.x += c.x;
                acc.y += c.y;
                acc.z += c.z;
                ++have;
            }
        }
    }
    if (have > 0) {
        const float inv = 1.f / have;
        storeOut(out, pitchPx, x, y, make_float3(acc.x * inv, acc.y * inv, acc.z * inv), outScale);
    } else {
        storeOut(out, pitchPx, x, y, make_float3(0.f, 0.f, 0.f), 1.f);
    }
    (void)idx;
}

// Settled path, step 1: resolve every subsample once into a ShadeInput.
__global__ void buildCacheKernel(const FieldSample* __restrict__ field, int fw, int fh, int mx,
                                 int my, ShadeInput* __restrict__ cache, int w, int h,
                                 ShadeParams p, CompositeMap map) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const double px = (double)x - 0.5 * w, py = (double)y - 0.5 * h;
    const int ss = map.ss < 1 ? 1 : map.ss;
    ShadeInput* dst = cache + ((size_t)y * w + x) * (size_t)(ss * ss);
    for (int j = 0; j < ss; ++j) {
        for (int i = 0; i < ss; ++i) {
            const double sx = px + (i + 0.5) / ss - 0.5, sy = py + (j + 0.5) / ss - 0.5;
            FieldSample s;
            int xi, yi, gi;
            ShadeInput in{};
            if (resolveBest(field, fw, fh, mx, my, sx, sy, map, s, xi, yi, gi)) {
                in = prepareSample(s, p, gradientAt(field, fw, fh, xi, yi, s, p, map.gens[gi]));
            }
            dst[j * ss + i] = in;
        }
    }
}

// Settled path, step 2: colour the cache. Pure streaming.
__global__ void shadeCachedKernel(const ShadeInput* __restrict__ cache, uint16_t* __restrict__ out,
                                  int pitchPx, float outScale, int w, int h, int ss, ShadeParams p,
                                  float timeSec, const float4* __restrict__ lut) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t idx = (size_t)y * w + x;
    const ShadeInput* src = cache + idx * (size_t)(ss * ss);
    float3 acc = make_float3(0.f, 0.f, 0.f);
    for (int k = 0; k < ss * ss; ++k) {
        const float3 c = colourInput(src[k], p, timeSec, lut);
        acc.x += c.x;
        acc.y += c.y;
        acc.z += c.z;
    }
    const float inv = 1.f / (ss * ss);
    storeOut(out, pitchPx, x, y, make_float3(acc.x * inv, acc.y * inv, acc.z * inv), outScale);
}

// Settled path with a reconstruction filter, step 2a: colour every
// subsample into a field-resolution buffer.
__global__ void shadeSubKernel(const ShadeInput* __restrict__ cache, uint16_t* __restrict__ sub,
                               int w, int h, int ss, ShadeParams p, float timeSec,
                               const float4* __restrict__ lut) {
    const int fx = blockIdx.x * blockDim.x + threadIdx.x;
    const int fy = blockIdx.y * blockDim.y + threadIdx.y;
    const int vw = w * ss, vh = h * ss;
    if (fx >= vw || fy >= vh) return;
    const int X = fx / ss, Y = fy / ss, i = fx - X * ss, j = fy - Y * ss;
    const ShadeInput in = cache[((size_t)Y * w + X) * (size_t)(ss * ss) + j * ss + i];
    const float3 c = colourInput(in, p, timeSec, lut);
    ushort4 v;
    v.x = f2h(c.x);
    v.y = f2h(c.y);
    v.z = f2h(c.z);
    v.w = f2h(1.f);
    *reinterpret_cast<ushort4*>(sub + ((size_t)fy * vw + fx) * 4) = v;
}

// Step 2b: gather subsamples within the filter radius of each pixel centre,
// using their true (jittered) positions.
__global__ void filterKernel(const uint16_t* __restrict__ sub, uint16_t* __restrict__ out, int w,
                             int h, int ss, int mx, int my, int pattern, int jox, int joy,
                             int filter, float radius) {
    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const int vw = w * ss, vh = h * ss;
    const int R = (int)ceilf(radius - 0.5f) + 1;  // neighbour pixels to visit
    float3 acc = make_float3(0.f, 0.f, 0.f);
    float wsum = 0.f;
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            const int PX = X + dx, PY = Y + dy;
            if (PX < 0 || PX >= w || PY < 0 || PY >= h) continue;
            for (int j = 0; j < ss; ++j) {
                for (int i = 0; i < ss; ++i) {
                    const int fx = PX * ss + i, fy = PY * ss + j;
                    const float2 jit = sampleJitter(pattern, ss, mx + fx, my + fy, i, j, jox, joy);
                    // Position relative to this pixel's centre, in display pixels.
                    const float px = dx + (i + 0.5f + jit.x) / ss - 0.5f;
                    const float py = dy + (j + 0.5f + jit.y) / ss - 0.5f;
                    const float d = sqrtf(px * px + py * py);
                    const float wgt = filterWeight(filter, d, radius);
                    if (wgt <= 0.f) continue;
                    const ushort4 v = *reinterpret_cast<const ushort4*>(sub + ((size_t)fy * vw + fx) * 4);
                    acc.x += h2f(v.x) * wgt;
                    acc.y += h2f(v.y) * wgt;
                    acc.z += h2f(v.z) * wgt;
                    wsum += wgt;
                }
            }
        }
    }
    if (wsum <= 0.f) {
        // Radius too small to reach any sample: fall back to the pixel's own
        // subsamples, box averaged.
        acc = make_float3(0.f, 0.f, 0.f);
        for (int j = 0; j < ss; ++j)
            for (int i = 0; i < ss; ++i) {
                const ushort4 v = *reinterpret_cast<const ushort4*>(
                    sub + ((size_t)(Y * ss + j) * vw + (X * ss + i)) * 4);
                acc.x += h2f(v.x);
                acc.y += h2f(v.y);
                acc.z += h2f(v.z);
            }
        wsum = (float)(ss * ss);
    }
    const float inv = 1.f / wsum;
    ushort4 o;
    o.x = f2h(acc.x * inv);
    o.y = f2h(acc.y * inv);
    o.z = f2h(acc.z * inv);
    o.w = f2h(1.f);
    *reinterpret_cast<ushort4*>(out + ((size_t)Y * w + X) * 4) = o;
}

__global__ void paletteLutKernel(float4* __restrict__ lut, ShadeParams p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= PALETTE_LUT_SIZE) return;
    const float3 c = paletteDirect(p, (i + 0.5f) / PALETTE_LUT_SIZE);
    lut[i] = make_float4(c.x, c.y, c.z, 0.f);
}

}  // namespace

CudaRenderer::~CudaRenderer() {
    freeOutput();
    freeField();
    freeReference();
    freeBla();
    if (evStart_) cudaEventDestroy((cudaEvent_t)evStart_);
    if (evStop_) cudaEventDestroy((cudaEvent_t)evStop_);
    if (evSlice_) cudaEventDestroy((cudaEvent_t)evSlice_);
    if (evShadeA_) cudaEventDestroy((cudaEvent_t)evShadeA_);
    if (evShadeB_) cudaEventDestroy((cudaEvent_t)evShadeB_);
    if (activeCount_) cudaFree(activeCount_);
    if (minIter_) cudaFree(minIter_);
    minIter_ = nullptr;
    if (levelCount_) cudaFree(levelCount_);
    if (levelCountHost_) cudaFreeHost(levelCountHost_);
    if (paletteLut_) cudaFree(paletteLut_);
    if (sliceStart_ns_) cudaFree(sliceStart_ns_);
    if (activeCountHost_) cudaFreeHost(activeCountHost_);
    if (minIterHost_) cudaFreeHost(minIterHost_);
    minIterHost_ = nullptr;
    if (stream_) cudaStreamDestroy((cudaStream_t)stream_);
    if (dispStream_) cudaStreamDestroy((cudaStream_t)dispStream_);
}

bool CudaRenderer::init() {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::snprintf(deviceName_, sizeof deviceName_, "%s (SM %d.%d, %d SMs)", prop.name,
                  prop.major, prop.minor, prop.multiProcessorCount);
    cudaEvent_t a, b, es, sa, sb;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventCreateWithFlags(&es, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&sa));
    CUDA_CHECK(cudaEventCreate(&sb));
    evStart_ = a;
    evStop_ = b;
    evSlice_ = es;
    evShadeA_ = sa;
    evShadeB_ = sb;
    cudaStream_t st, ds;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ds, cudaStreamNonBlocking));
    stream_ = st;
    dispStream_ = ds;
    CUDA_CHECK(cudaMalloc(&activeCount_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&minIter_, sizeof(int)));
    CUDA_CHECK(cudaMemset(minIter_, 0x7f, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&levelCount_, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&levelCountHost_, sizeof(int)));
    *levelCountHost_ = 0;
    CUDA_CHECK(cudaMalloc(&sliceStart_ns_, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocHost(&activeCountHost_, sizeof(int)));
    *activeCountHost_ = 0;
    CUDA_CHECK(cudaMallocHost(&minIterHost_, sizeof(int)));
    *minIterHost_ = 0x7f7f7f7f;
    minIterGen_ = -1.f;
    return true;
}

void CudaRenderer::syncAll() {
    if (stream_) cudaStreamSynchronize((cudaStream_t)stream_);
    if (dispStream_) cudaStreamSynchronize((cudaStream_t)dispStream_);
}

void CudaRenderer::freeOutput() {
    syncAll();
    if (out_) {
        cudaFree(out_);
        out_ = nullptr;
    }
    if (extMem_) {
        cudaDestroyExternalMemory(extMem_);
        extMem_ = nullptr;
    }
    outPitchPx_ = 0;
}

bool CudaRenderer::syncDisplay() {
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)dispStream_));
    return true;
}

bool CudaRenderer::readOutputLinear(std::vector<float>& rgb) {
    if (!out_) return false;
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)dispStream_));
    const size_t n = (size_t)outPitchPx_ * height_ * 4;
    std::vector<uint16_t> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), out_, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    rgb.resize((size_t)width_ * height_ * 3);
    auto halfToFloat = [](uint16_t v) {
        const uint32_t sgn = (v >> 15) & 1u, exp = (v >> 10) & 0x1Fu, man = v & 0x3FFu;
        float f;
        if (exp == 0) f = std::ldexp((float)man, -24);
        else if (exp == 31) f = man ? 0.f : 1e30f;
        else f = std::ldexp((float)(man | 0x400u), (int)exp - 25);
        return sgn ? -f : f;
    };
    const float inv = outScale_ > 0.f ? 1.f / outScale_ : 1.f;
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const uint16_t* px = h.data() + ((size_t)y * outPitchPx_ + x) * 4;
            float* o = rgb.data() + ((size_t)y * width_ + x) * 3;
            o[0] = halfToFloat(px[0]) * inv;
            o[1] = halfToFloat(px[1]) * inv;
            o[2] = halfToFloat(px[2]) * inv;
        }
    }
    return true;
}

bool CudaRenderer::readOutput(std::vector<uint32_t>& rgba8) {
    if (!out_) return false;
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)dispStream_));
    const size_t n = (size_t)outPitchPx_ * height_ * 4;
    std::vector<uint16_t> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), out_, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    rgba8.resize((size_t)width_ * height_);
    auto halfToFloat = [](uint16_t v) {
        const uint32_t sgn = (v >> 15) & 1u, exp = (v >> 10) & 0x1Fu, man = v & 0x3FFu;
        float f;
        if (exp == 0) f = std::ldexp((float)man, -24);
        else if (exp == 31) f = man ? 0.f : 1e30f;
        else f = std::ldexp((float)(man | 0x400u), (int)exp - 25);
        return sgn ? -f : f;
    };
    auto enc = [](float v) {
        v = std::min(std::max(v, 0.f), 1.f);
        v = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
        return (uint32_t)(v * 255.f + 0.5f);
    };
    const float inv = outScale_ > 0.f ? 1.f / outScale_ : 1.f;
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const uint16_t* px = h.data() + ((size_t)y * outPitchPx_ + x) * 4;
            const uint32_t r = enc(halfToFloat(px[0]) * inv), g = enc(halfToFloat(px[1]) * inv),
                           b = enc(halfToFloat(px[2]) * inv);
            rgba8[(size_t)y * width_ + x] = r | (g << 8) | (b << 16) | 0xFF000000u;
        }
    }
    return true;
}

void CudaRenderer::freeField() {
    syncAll();
    if (field_) cudaFree(field_);
    if (state_) cudaFree(state_);
    if (fieldAlt_) cudaFree(fieldAlt_);
    if (stateAlt_) cudaFree(stateAlt_);
    if (shadeCache_) cudaFree(shadeCache_);
    if (hdr_) cudaFree(hdr_);
    if (subColour_) cudaFree(subColour_);
    subColour_ = nullptr;
    if (bloomA_) cudaFree(bloomA_);
    if (bloomT_) cudaFree(bloomT_);
    if (bloomB_) cudaFree(bloomB_);
    if (bloomT2_) cudaFree(bloomT2_);
    field_ = fieldAlt_ = nullptr;
    state_ = stateAlt_ = nullptr;
    shadeCache_ = nullptr;
    hdr_ = nullptr;
    bloomA_ = bloomT_ = bloomB_ = bloomT2_ = nullptr;
    iterDone_ = true;
    sliceInFlight_ = false;
}

void CudaRenderer::freeReference() {
    syncAll();
    if (refZr_) cudaFree(refZr_);
    if (refZi_) cudaFree(refZi_);
    if (refF_) cudaFree(refF_);
    refZr_ = refZi_ = nullptr;
    refF_ = nullptr;
    refCapacity_ = 0;
    ref_ = DeviceReference{};
}

void CudaRenderer::freeBla() {
    syncAll();
    if (blaNodes_) cudaFree(blaNodes_);
    if (blaNodesF_) cudaFree(blaNodesF_);
    if (blaOffsets_) cudaFree(blaOffsets_);
    blaNodes_ = nullptr;
    blaNodesF_ = nullptr;
    blaOffsets_ = nullptr;
    blaNodeCapacity_ = blaLevelCapacity_ = 0;
    bla_ = DeviceBla{};
}

bool CudaRenderer::uploadBla(const BlaNode* nodes, const BlaNodeF* nodesF, int count,
                             const int* levelOffset, int levels, int steps) {
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream_));
    if (count > blaNodeCapacity_ || levels > blaLevelCapacity_) {
        freeBla();
        blaNodeCapacity_ = count + count / 4 + 1024;
        blaLevelCapacity_ = levels + 8;
        CUDA_CHECK(cudaMalloc(&blaNodes_, sizeof(BlaNode) * (size_t)blaNodeCapacity_));
        CUDA_CHECK(cudaMalloc(&blaNodesF_, sizeof(BlaNodeF) * (size_t)blaNodeCapacity_));
        CUDA_CHECK(cudaMalloc(&blaOffsets_, sizeof(int) * (size_t)blaLevelCapacity_));
    }
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(blaNodes_, nodes, sizeof(BlaNode) * (size_t)count,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(blaNodesF_, nodesF, sizeof(BlaNodeF) * (size_t)count,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(blaOffsets_, levelOffset, sizeof(int) * (size_t)levels,
                              cudaMemcpyHostToDevice));
    }
    bla_.nodes = blaNodes_;
    bla_.levelOffset = blaOffsets_;
    bla_.levels = levels;
    bla_.steps = steps;
    return true;
}

bool CudaRenderer::uploadReference(const double* zr, const double* zi, int length, bool escaped) {
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream_));  // no slice may read the old orbit
    if (length > refCapacity_) {
        freeReference();
        refCapacity_ = length + length / 4 + 1024;
        CUDA_CHECK(cudaMalloc(&refZr_, sizeof(double) * (size_t)refCapacity_));
        CUDA_CHECK(cudaMalloc(&refZi_, sizeof(double) * (size_t)refCapacity_));
        CUDA_CHECK(cudaMalloc(&refF_, sizeof(float2) * (size_t)refCapacity_));
    }
    CUDA_CHECK(cudaMemcpy(refZr_, zr, sizeof(double) * (size_t)length, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(refZi_, zi, sizeof(double) * (size_t)length, cudaMemcpyHostToDevice));
    {
        std::vector<float2> f((size_t)length);
        for (int i = 0; i < length; ++i) f[(size_t)i] = make_float2((float)zr[i], (float)zi[i]);
        CUDA_CHECK(cudaMemcpy(refF_, f.data(), sizeof(float2) * (size_t)length,
                              cudaMemcpyHostToDevice));
    }
    ref_.zr = refZr_;
    ref_.zi = refZi_;
    ref_.length = length;
    ref_.escaped = escaped;
    return true;
}

bool CudaRenderer::bindOutput(void* sharedHandle, size_t sharedSize, int rowPitchBytes, int width,
                              int height, int ss) {
    freeOutput();
    freeField();
    width_ = width;
    height_ = height;
    ss_ = ss < 1 ? 1 : ss;
    if (!sharedHandle || width <= 0 || height <= 0) return true;
    {
        cudaExternalMemoryHandleDesc hd{};
        hd.type = cudaExternalMemoryHandleTypeD3D12Resource;
        hd.handle.win32.handle = sharedHandle;
        hd.size = sharedSize;
        hd.flags = cudaExternalMemoryDedicated;
        CUDA_CHECK(cudaImportExternalMemory(&extMem_, &hd));
        cudaExternalMemoryBufferDesc bd{};
        bd.offset = 0;
        bd.size = sharedSize;
        bd.flags = 0;
        void* ptr = nullptr;
        CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&ptr, extMem_, &bd));
        out_ = (uint16_t*)ptr;
        outPitchPx_ = rowPitchBytes / 8;
    }
    viewW_ = width * ss_;
    viewH_ = height * ss_;
    marginX_ = (viewW_ + 7) / 8;
    marginY_ = (viewH_ + 7) / 8;
    fieldW_ = viewW_ + 2 * marginX_;
    fieldH_ = viewH_ + 2 * marginY_;
    const size_t n = (size_t)width * height;
    const size_t fn = (size_t)fieldW_ * fieldH_;
    CUDA_CHECK(cudaMalloc(&field_, sizeof(FieldSample) * fn));
    CUDA_CHECK(cudaMalloc(&state_, sizeof(PixelState) * fn));
    CUDA_CHECK(cudaMalloc(&fieldAlt_, sizeof(FieldSample) * fn));
    CUDA_CHECK(cudaMalloc(&stateAlt_, sizeof(PixelState) * fn));
    CUDA_CHECK(cudaMalloc(&shadeCache_, sizeof(ShadeInput) * n * (size_t)(ss_ * ss_)));
    CUDA_CHECK(cudaMalloc(&hdr_, sizeof(uint16_t) * 4 * n));
    CUDA_CHECK(cudaMalloc(&subColour_, sizeof(uint16_t) * 4 * (size_t)viewW_ * viewH_));
    jitterOx_ = jitterOy_ = 0;
    {
        const size_t aw = (width + 3) / 4, ah = (height + 3) / 4;
        const size_t bw = (aw + 1) / 2, bh = (ah + 1) / 2;
        CUDA_CHECK(cudaMalloc(&bloomA_, sizeof(float4) * aw * ah));
        CUDA_CHECK(cudaMalloc(&bloomT_, sizeof(float4) * aw * ah));
        CUDA_CHECK(cudaMalloc(&bloomB_, sizeof(float4) * bw * bh));
        CUDA_CHECK(cudaMalloc(&bloomT2_, sizeof(float4) * bw * bh));
    }
    if (!paletteLut_) CUDA_CHECK(cudaMalloc(&paletteLut_, sizeof(float4) * PALETTE_LUT_SIZE));
    clearField<<<(int)((fn + 255) / 256), 256>>>(field_, (int)fn);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

void CudaRenderer::resetPass(const ViewParams& view) {
    iterView_ = view;
    bla_.enabled = view.useBla ? 1 : 0;
    sliceStart_ = 0;
    sliceIters_ = view.maxIter;  // the GPU-clock deadline bounds each launch
    sliceInFlight_ = false;
    stride_ = firstStride_;
    completedStride_ = 0;
    iterDone_ = false;
    passMs_ = 0.f;
    for (int i = 0; i < kLevels; ++i) {
        levelPixels_[i] = -1;
        levelMs_[i] = 0.f;
        levelDone_[i] = false;
    }
    levelCountPending_ = false;
    lastActive_ = 0;
}

int CudaRenderer::progress_levelIndex() const {
    return stride_ >= 8 ? 0 : stride_ >= 4 ? 1 : stride_ >= 2 ? 2 : 3;
}

float CudaRenderer::progress() const {
    if (iterDone_) return 1.f;
    double doneMs = 0.0, donePx = 0.0;
    for (int i = 0; i < kLevels; ++i) {
        if (levelDone_[i] && levelPixels_[i] > 0) {
            doneMs += levelMs_[i];
            donePx += levelPixels_[i];
        }
    }
    const int cur = progress_levelIndex();
    // Pixels of the current and future levels. Unknown counts are estimated
    // from geometry (a full pass) as a fallback.
    double futurePx = 0.0;
    for (int i = cur; i < kLevels; ++i) {
        double n = levelPixels_[i];
        if (n < 0) {
            const int st = 8 >> i;
            if (st == 1) n = (double)viewW_ * viewH_ * 0.75;
            else n = ((double)fieldW_ / st) * ((double)fieldH_ / st) * 0.75;
        }
        futurePx += n;
    }
    if (donePx > 0.0) {
        const double c = doneMs / donePx;  // ms per pixel, from completed levels
        const double total = doneMs + c * futurePx;
        return (float)std::min(passMs_ / std::max(total, 1e-3), 0.995);
    }
    // Nothing completed yet: count-based fraction of the first level, scaled
    // to that level's share of the pass.
    const double n0 = levelPixels_[cur] > 0 ? levelPixels_[cur] : 1.0;
    const double frac = std::max(0.0, 1.0 - (double)lastActive_ / n0);
    return (float)std::min(frac * n0 / std::max(futurePx, 1.0), 0.995);
}

float CudaRenderer::etaMs() const {
    if (iterDone_) return 0.f;
    const float p = progress();
    if (p <= 0.001f) return -1.f;  // unknown
    return std::max(passMs_ / p - passMs_, 0.f);
}

bool CudaRenderer::beginIterate(const ViewParams& view, int gen) {
    if (!field_ || !state_ || view.width != viewW_ || view.height != viewH_) return false;
    if (ref_.length < 2) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    CUDA_CHECK(cudaStreamSynchronize(stream));  // cancel the slice in flight
    resetPass(view);
    gen_ = gen;
    minIterGen_ = -1.f;
    CUDA_CHECK(cudaMemsetAsync(minIter_, 0x7f, sizeof(int), stream));  // "none yet"
    const int count = fieldW_ * fieldH_;
    resetState<<<(count + 255) / 256, 256, 0, stream>>>(state_, count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool CudaRenderer::shiftAndResume(const ViewParams& view, int dx, int dy) {
    if (!field_ || !state_ || view.width != viewW_ || view.height != viewH_) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    syncAll();  // the composite may be reading the buffers we are about to swap
    if (dx != 0 || dy != 0) {
        const dim3 block(32, 8);
        const dim3 grid((fieldW_ + block.x - 1) / block.x, (fieldH_ + block.y - 1) / block.y);
        shiftKernel<<<grid, block, 0, stream>>>(state_, field_, stateAlt_, fieldAlt_, fieldW_,
                                                fieldH_, dx, dy);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::swap(state_, stateAlt_);
        std::swap(field_, fieldAlt_);
        // A sample at index x + dx now sits at x: keep its jitter key.
        jitterOx_ += dx;
        jitterOy_ += dy;
    }
    const float keepPass = passMs_;
    resetPass(view);
    passMs_ = keepPass;
    return true;
}

bool CudaRenderer::restartPending(const ViewParams& view) {
    if (!field_ || !state_) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int count = fieldW_ * fieldH_;
    resetPendingState<<<(count + 255) / 256, 256, 0, stream>>>(state_, count,
                                                               iterView_.useFloat ? 8 : 10);
    CUDA_CHECK(cudaGetLastError());
    const float keepPass = passMs_;
    resetPass(view);
    passMs_ = keepPass;
    return true;
}

bool CudaRenderer::iterateBusy() {
    if (!sliceInFlight_) return false;
    const cudaError_t q = cudaEventQuery((cudaEvent_t)evSlice_);
    if (q == cudaErrorNotReady) return true;
    // Slice finished: collect timing and the active count.
    sliceInFlight_ = false;
    sliceFinished_ = true;
    cudaEventElapsedTime(&sliceMs_, (cudaEvent_t)evStart_, (cudaEvent_t)evStop_);
    passMs_ += sliceMs_;
    iterateMs_ = passMs_;
    {
        const int li = progress_levelIndex();
        levelMs_[li] += sliceMs_;
        if (levelCountPending_) {
            levelPixels_[li] = *levelCountHost_;
            levelCountPending_ = false;
        }
        lastActive_ = *activeCountHost_;
        if (*minIterHost_ != 0x7f7f7f7f) {
            float m;
            std::memcpy(&m, minIterHost_, sizeof m);
            minIterGen_ = m;
        }
    }
    if (*activeCountHost_ == 0) {
        completedStride_ = stride_;
        levelDone_[progress_levelIndex()] = true;
        if (stride_ <= 1) {
            iterDone_ = true;
        } else {
            stride_ /= 2;
            sliceStart_ = 0;
        }
    }
    return false;
}

bool CudaRenderer::stepIterate() {
    if (iterDone_ || sliceInFlight_) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    const dim3 block(32, 8);
    int sw, sh;
    if (stride_ == 1) {
        sw = viewW_;
        sh = viewH_;
    } else {
        sw = (fieldW_ + stride_ - 1) / stride_;
        sh = (fieldH_ + stride_ - 1) / stride_;
    }
    const dim3 grid((sw + block.x - 1) / block.x, (sh + block.y - 1) / block.y);
    CUDA_CHECK(cudaMemsetAsync(activeCount_, 0, sizeof(int), stream));
    if (levelPixels_[progress_levelIndex()] < 0 && !levelCountPending_) {
        // First slice of this level: count what it has to do.
        CUDA_CHECK(cudaMemsetAsync(levelCount_, 0, sizeof(int), stream));
        countLevelKernel<<<grid, block, 0, stream>>>(state_, fieldW_, fieldH_, marginX_, marginY_,
                                                     viewW_, viewH_, stride_,
                                                     iterView_.useFloat ? 8 : 10, levelCount_);
        CUDA_CHECK(cudaMemcpyAsync(levelCountHost_, levelCount_, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        levelCountPending_ = true;
    }
    cudaEventRecord((cudaEvent_t)evStart_, stream);
    stampTimer<<<1, 1, 0, stream>>>(sliceStart_ns_);
    if (iterView_.useFloat) {
        int eP;
        const double mP = std::frexp(iterView_.scale, &eP);
        DeviceReferenceF rf;
        rf.z = refF_;
        rf.length = ref_.length;
        DeviceBlaF bf;
        bf.nodes = blaNodesF_;
        bf.levelOffset = bla_.levelOffset;
        bf.levels = bla_.levels;
        bf.steps = bla_.steps;
        bf.enabled = bla_.enabled;
        iterateSliceF<<<grid, block, 0, stream>>>(
            (PixelStateF*)state_, field_, fieldW_, fieldH_, marginX_, marginY_, viewW_, viewH_,
            stride_, (float)mP, eP, iterView_.refOffX / iterView_.scale,
            iterView_.refOffY / iterView_.scale, iterView_.maxIter, sliceIters_, sliceStart_ns_,
            kSliceBudgetNs, rf, bf, gen_, ss_, iterView_.aaPattern, jitterOx_, jitterOy_,
            activeCount_, minIter_);
    } else {
        iterateSlice<<<grid, block, 0, stream>>>(
            state_, field_, fieldW_, fieldH_, marginX_, marginY_, viewW_, viewH_, stride_,
            iterView_.scale, iterView_.refOffX, iterView_.refOffY, iterView_.maxIter, sliceIters_,
            sliceStart_ns_, kSliceBudgetNs, ref_, bla_, gen_, ss_, iterView_.aaPattern, jitterOx_,
            jitterOy_, activeCount_, minIter_);
    }
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord((cudaEvent_t)evStop_, stream);
    CUDA_CHECK(cudaMemcpyAsync(activeCountHost_, activeCount_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(minIterHost_, minIter_, sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    cudaEventRecord((cudaEvent_t)evSlice_, stream);
    sliceStart_ = iterView_.maxIter;
    sliceInFlight_ = true;
    return true;
}

bool CudaRenderer::debugStats(int gen, DebugStats& out) {
    out = DebugStats{};
    if (!field_ || !state_) return false;
    syncAll();
    const size_t fn = (size_t)fieldW_ * fieldH_;
    std::vector<FieldSample> f(fn);
    std::vector<PixelState> st(fn);
    CUDA_CHECK(cudaMemcpy(f.data(), field_, sizeof(FieldSample) * fn, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(st.data(), state_, sizeof(PixelState) * fn, cudaMemcpyDeviceToHost));
    for (int y = marginY_; y < marginY_ + viewH_; ++y) {
        for (int x = marginX_; x < marginX_ + viewW_; ++x) {
            const size_t i = (size_t)y * fieldW_ + x;
            ++out.total;
            if ((int)f[i].gen == gen) {
                ++out.genMatch;
                if (f[i].iter < 0.f) ++out.genMatchInside;
            } else if (f[i].gen == 0) {
                ++out.genZero;
            } else {
                ++out.genOther;
            }
            const int status = iterView_.useFloat
                                   ? reinterpret_cast<const PixelStateF*>(st.data())[i].status
                                   : st[i].status;
            if (status == 0) ++out.stActive;
            else if (status == 1) ++out.stEscaped;
            else ++out.stInside;
        }
    }
    return true;
}

bool CudaRenderer::composite(const ShadeParams& params, float timeSec, const CompositeMap& map) {
    if (!hdr_ || !field_) return false;
    cudaStream_t stream = (cudaStream_t)dispStream_;
    const dim3 block(16, 16);
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    // Timing of the previous display pass, harvested without blocking.
    if (shadePending_ && cudaEventQuery((cudaEvent_t)evShadeB_) == cudaSuccess) {
        cudaEventElapsedTime(&shadeMs_, (cudaEvent_t)evShadeA_, (cudaEvent_t)evShadeB_);
        shadePending_ = false;
    }
    if (!shadePending_) cudaEventRecord((cudaEvent_t)evShadeA_, stream);
    compositeKernel<<<grid, block, 0, stream>>>(field_, fieldW_, fieldH_, marginX_, marginY_, hdr_,
                                                width_, 1.f, width_, height_, params, timeSec, map,
                                                paletteLut_);
    const cudaError_t err = cudaGetLastError();
    if (!shadePending_) {
        cudaEventRecord((cudaEvent_t)evShadeB_, stream);
        shadePending_ = true;
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "composite kernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool CudaRenderer::buildPaletteLut(const ShadeParams& params) {
    if (!paletteLut_) return false;
    cudaStream_t stream = (cudaStream_t)dispStream_;
    paletteLutKernel<<<PALETTE_LUT_SIZE / 256, 256, 0, stream>>>(paletteLut_, params);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool CudaRenderer::buildShadeCache(const ShadeParams& params, const CompositeMap& map) {
    if (!field_ || !shadeCache_) return false;
    cudaStream_t stream = (cudaStream_t)dispStream_;
    const dim3 block(16, 16);
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    buildCacheKernel<<<grid, block, 0, stream>>>(field_, fieldW_, fieldH_, marginX_, marginY_,
                                                 shadeCache_, width_, height_, params, map);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool CudaRenderer::shadeCached(const ShadeParams& params, float timeSec, const AaParams& aa) {
    if (!hdr_ || !shadeCache_) return false;
    cudaStream_t stream = (cudaStream_t)dispStream_;
    const dim3 block(16, 16);
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    if (shadePending_ && cudaEventQuery((cudaEvent_t)evShadeB_) == cudaSuccess) {
        cudaEventElapsedTime(&shadeMs_, (cudaEvent_t)evShadeA_, (cudaEvent_t)evShadeB_);
        shadePending_ = false;
    }
    if (!shadePending_) cudaEventRecord((cudaEvent_t)evShadeA_, stream);
    if (aa.filter == FILTER_BOX) {
        shadeCachedKernel<<<grid, block, 0, stream>>>(shadeCache_, hdr_, width_, 1.f, width_,
                                                      height_, ss_, params, timeSec, paletteLut_);
    } else {
        const dim3 gsub((viewW_ + block.x - 1) / block.x, (viewH_ + block.y - 1) / block.y);
        shadeSubKernel<<<gsub, block, 0, stream>>>(shadeCache_, subColour_, width_, height_, ss_,
                                                   params, timeSec, paletteLut_);
        filterKernel<<<grid, block, 0, stream>>>(subColour_, hdr_, width_, height_, ss_, marginX_,
                                                 marginY_, aa.pattern, jitterOx_, jitterOy_,
                                                 aa.filter, fmaxf(aa.radius, 0.05f));
    }
    const cudaError_t err = cudaGetLastError();
    if (!shadePending_) {
        cudaEventRecord((cudaEvent_t)evShadeB_, stream);
        shadePending_ = true;
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "shadeCached kernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool CudaRenderer::postProcess(const PostParams& params, uint32_t frame) {
    if (!out_ || !hdr_) return false;
    cudaStream_t stream = (cudaStream_t)dispStream_;
    const dim3 block(16, 16);
    const int aw = (width_ + 3) / 4, ah = (height_ + 3) / 4;
    const int bw = (aw + 1) / 2, bh = (ah + 1) / 2;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0, stream);
    if (params.bloomIntensity > 0.f) {
        const dim3 ga((aw + 15) / 16, (ah + 15) / 16), gb((bw + 15) / 16, (bh + 15) / 16);
        postBloomDown<<<ga, block, 0, stream>>>(hdr_, width_, height_, bloomA_, aw, ah,
                                                params.bloomThreshold, params.bloomKnee);
        postBlur<<<ga, block, 0, stream>>>(bloomA_, bloomT_, aw, ah, params.bloomRadius, 1, 0);
        postBlur<<<ga, block, 0, stream>>>(bloomT_, bloomA_, aw, ah, params.bloomRadius, 0, 1);
        postDown2<<<gb, block, 0, stream>>>(bloomA_, aw, ah, bloomB_, bw, bh);
        postBlur<<<gb, block, 0, stream>>>(bloomB_, bloomT2_, bw, bh, params.bloomRadius, 1, 0);
        postBlur<<<gb, block, 0, stream>>>(bloomT2_, bloomB_, bw, bh, params.bloomRadius, 0, 1);
    }
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    postFinal<<<grid, block, 0, stream>>>(hdr_, width_, height_, bloomA_, aw, ah, bloomB_, bw, bh,
                                          out_, outPitchPx_, params, outScale_, headroom_, frame);
    const cudaError_t err = cudaGetLastError();
    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&postMs_, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "post kernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool CudaRenderer::findUnreliable(int& fx, int& fy) {
    if (!state_ || !levelCount_) return false;
    cudaStream_t stream = (cudaStream_t)stream_;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int count = fieldW_ * fieldH_;
    const int init = 0x7fffffff;
    CUDA_CHECK(cudaMemcpy(levelCount_, &init, sizeof(int), cudaMemcpyHostToDevice));
    findUnreliableKernel<<<(count + 255) / 256, 256, 0, stream>>>(state_, count, levelCount_);
    CUDA_CHECK(cudaGetLastError());
    int idx = init;
    CUDA_CHECK(cudaMemcpy(&idx, levelCount_, sizeof(int), cudaMemcpyDeviceToHost));
    if (idx == init) return false;
    fx = idx % fieldW_;
    fy = idx / fieldW_;
    return true;
}
