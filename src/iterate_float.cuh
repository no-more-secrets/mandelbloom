#pragma once
// Float perturbation kernel. Included by render_cuda.cu only.
//
// Every small quantity is a float mantissa with its own power-of-two
// exponent ("floatexp"): the delta dz = wz * 2^ez, the derivative
// D = dz/dc = wd * 2^ed, the pixel size P = mP * 2^eP, and the BLA
// coefficients. Mantissas stay within [2^-32, 2^32] by renormalising, so
// zooms far past float range work at FP32 speed. The reference orbit is
// float2; that is standard (Kalles Fraktaler, Fraktaler 3) and adequate for
// escape times.
#include <cstdint>
#include "bla.h"
#include "field.h"

// Same size as PixelState (48 bytes) so the shared reset/shift kernels work.
struct PixelStateF {
    float wzr, wzi;  // dz mantissa
    int ez;
    float wdr, wdi;  // D mantissa
    int ed;
    int m;
    int n;
    int status;  // 0 active, 1 escaped, 2 inside
    int hint;    // BLA level used last time (search starts one above it)
    int pad[2];
};

struct DeviceReferenceF {
    const float2* z = nullptr;
    int length = 0;
};

struct DeviceBlaF {
    const BlaNodeF* nodes = nullptr;
    const int* levelOffset = nullptr;
    int levels = 0;
    int steps = 0;
    int enabled = 0;
};

namespace fx {

// Keep the larger component within [2^-32, 2^32). Tested on the component,
// not the squared magnitude, which would underflow long before that.
__device__ __forceinline__ void renorm(float& r, float& i, int& e) {
    const float a = fmaxf(fabsf(r), fabsf(i));
    if (a > 4294967296.f || (a < 2.3283064e-10f && a > 0.f)) {
        int k;
        frexpf(a, &k);
        r = ldexpf(r, -k);
        i = ldexpf(i, -k);
        e += k;
    }
}

// (r, i) * 2^e  +  (ar, ai) * 2^ea, result exponent max(e, ea): neither term
// is shifted up, so nothing can overflow.
__device__ __forceinline__ void addFx(float& r, float& i, int& e, float ar, float ai, int ea) {
    const int m = max(e, ea);
    r = ldexpf(r, e - m) + ldexpf(ar, ea - m);
    i = ldexpf(i, e - m) + ldexpf(ai, ea - m);
    e = m;
}

// Longest valid BLA node starting at reference index m, for |dz| = |wz| 2^ez.
__device__ __forceinline__ const BlaNodeF* findBla(const DeviceBlaF& bla, int m, float wzmag2,
                                                   int ez, int refLast, int& hint) {
    const int j0 = m - 1;
    if (j0 < 0 || j0 >= bla.steps) return nullptr;
    const int kmax = j0 == 0 ? bla.levels - 1 : min(bla.levels - 1, __ffs(j0) - 1);
    // Try one level above the last successful one first; if that is valid,
    // climb while valid, else descend.
    int k = min(kmax, hint + 1);
    const BlaNodeF* best = nullptr;
    for (; k >= 0; --k) {
        const BlaNodeF* n = bla.nodes + bla.levelOffset[k] + (j0 >> k);
        if (n->r2m > 0.f && m + n->l <= refLast &&
            ldexpf(wzmag2, 2 * ez - n->r2e) < n->r2m) {
            best = n;
            break;
        }
    }
    if (best && k == min(kmax, hint + 1)) {
        // Valid at the first probe: maybe a higher level is valid too.
        for (int k2 = k + 1; k2 <= kmax; ++k2) {
            const BlaNodeF* n = bla.nodes + bla.levelOffset[k2] + (j0 >> k2);
            if (n->r2m > 0.f && m + n->l <= refLast &&
                ldexpf(wzmag2, 2 * ez - n->r2e) < n->r2m) {
                best = n;
                k = k2;
            } else {
                break;
            }
        }
    }
    hint = best ? k : -1;
    return best;
}

}  // namespace fx

__global__ void iterateSliceF(PixelStateF* __restrict__ state, FieldSample* __restrict__ field,
                              int fw, int fh, int mx, int my, int w, int h, int stride,
                              float mP, int eP, double refOffPx, double refOffPy, int maxIter,
                              int sliceIters, const unsigned long long* __restrict__ startNs,
                              unsigned long long budgetNs, DeviceReferenceF ref, DeviceBlaF bla,
                              float gen, int* __restrict__ activeCount) {
    const unsigned long long deadlineNs = *startNs + budgetNs;
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
    const size_t idx = inBounds ? (size_t)y * fw + x : 0;
    PixelStateF st{};
    if (inBounds) st = state[idx];
    bool active = inBounds && st.status == 0;

    if (active) {
        // dc = (mP * poff) * 2^eP, poff in pixels (exact in float).
        const float poffx = (float)(refOffPx + (double)(x - mx) - 0.5 * w);
        const float poffy = (float)(refOffPy - ((double)(y - my) - 0.5 * h));
        const float dcr = mP * poffx, dci = mP * poffy;
        float wzr = st.wzr, wzi = st.wzi, wdr = st.wdr, wdi = st.wdi;
        int ez = st.ez, ed = st.ed;
        int m = st.m, n = st.n;
        int hint = st.hint;
        float zr = 0.f, zi = 0.f;
        const float bailout = 65536.f;
        const int refLast = ref.length - 1;
        const int stop = min(maxIter, n + sliceIters);
        bool escaped = false;
        int budgetCheck = 0;

        while (n < stop) {
            if ((++budgetCheck & 63) == 0 && globalTimerNs() > deadlineNs) break;
            bool stepped = false;
            if (bla.enabled) {
                const BlaNodeF* nd = fx::findBla(bla, m, wzr * wzr + wzi * wzi, ez, refLast, hint);
                if (nd) {
                    // dz' = A dz + B dc
                    {
                        float tr = nd->ar * wzr - nd->ai * wzi;
                        float ti = nd->ar * wzi + nd->ai * wzr;
                        int te = nd->ae + ez;
                        fx::addFx(tr, ti, te, nd->br * dcr - nd->bi * dci,
                                  nd->br * dci + nd->bi * dcr, nd->be + eP);
                        wzr = tr;
                        wzi = ti;
                        ez = te;
                        fx::renorm(wzr, wzi, ez);
                    }
                    // D' = A D + B
                    {
                        float ur = nd->ar * wdr - nd->ai * wdi;
                        float ui = nd->ar * wdi + nd->ai * wdr;
                        int ue = nd->ae + ed;
                        fx::addFx(ur, ui, ue, nd->br, nd->bi, nd->be);
                        wdr = ur;
                        wdi = ui;
                        ed = ue;
                        fx::renorm(wdr, wdi, ed);
                    }
                    m += nd->l;
                    n += nd->l;
                    stepped = true;
                }
            }
            if (!stepped) {
                const float2 Z = ref.z[m];
                const float dzr = ldexpf(wzr, ez), dzi = ldexpf(wzi, ez);
                const float fzr = Z.x + dzr, fzi = Z.y + dzi;
                // D' = 2 z D + 1
                {
                    float nr = 2.f * (fzr * wdr - fzi * wdi);
                    float ni = 2.f * (fzr * wdi + fzi * wdr);
                    int ne = ed;
                    fx::addFx(nr, ni, ne, 1.f, 0.f, 0);
                    wdr = nr;
                    wdi = ni;
                    ed = ne;
                    fx::renorm(wdr, wdi, ed);
                }
                // dz' = (2Z + dz) dz + dc
                {
                    const float ar = 2.f * Z.x + dzr, ai = 2.f * Z.y + dzi;
                    float nr = ar * wzr - ai * wzi;
                    float ni = ar * wzi + ai * wzr;
                    int ne = ez;
                    fx::addFx(nr, ni, ne, dcr, dci, eP);
                    wzr = nr;
                    wzi = ni;
                    ez = ne;
                    fx::renorm(wzr, wzi, ez);
                }
                ++m;
                ++n;
            }
            const float2 Zn = ref.z[m];
            zr = Zn.x + ldexpf(wzr, ez);
            zi = Zn.y + ldexpf(wzi, ez);
            const float zmag2 = zr * zr + zi * zi;
            if (zmag2 > bailout) {
                escaped = true;
                break;
            }
            const float dzmag2 = ldexpf(wzr * wzr + wzi * wzi, 2 * ez);
            if (zmag2 < dzmag2 || m >= refLast) {
                wzr = zr;
                wzi = zi;
                ez = 0;
                fx::renorm(wzr, wzi, ez);
                m = 0;
            }
        }
        st.wzr = wzr;
        st.wzi = wzi;
        st.ez = ez;
        st.wdr = wdr;
        st.wdi = wdi;
        st.ed = ed;
        st.m = m;
        st.n = n;
        st.hint = hint;
        if (escaped) st.status = 1;
        else if (n >= maxIter) st.status = 2;
        const bool pending = st.status == 0;
        if (!pending) {
            FieldSample s{};
            if (!escaped) {
                s.iter = -1.f;
            } else {
                const float mag2 = zr * zr + zi * zi;
                const float logMag = 0.5f * logf(mag2);
                s.iter = (float)n + 1.f - log2f(logMag / 0.69314718f);
                const float wdmag = sqrtf(wdr * wdr + wdi * wdi);
                // DE in pixels: |z| log|z| / |D| / P
                if (wdmag > 0.f) {
                    s.de = ldexpf(sqrtf(mag2) * logMag / (wdmag * mP), -ed - eP);
                    // Milnor normal: direction of z / D = z * conj(D)
                    const float ur = zr * wdr + zi * wdi, ui = zi * wdr - zr * wdi;
                    const float um = sqrtf(ur * ur + ui * ui);
                    if (um > 0.f) {
                        s.nx = ur / um;
                        s.ny = ui / um;
                    }
                }
                s.angle = atan2f(zi, zr);
            }
            s.gen = gen;
            field[idx] = s;
        }
        state[idx] = st;
        active = pending;
    }

    const unsigned mask = __ballot_sync(0xffffffffu, active);
    if ((threadIdx.x & 31) == 0 && mask) atomicAdd(activeCount, __popc(mask));
}
