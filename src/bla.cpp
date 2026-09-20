#include "bla.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cmath>

namespace {

inline double cabs(double r, double i) { return std::sqrt(r * r + i * i); }

// Merge x (applied first) then y.
BlaNode merge(const BlaNode& x, const BlaNode& y, double cMax) {
    BlaNode z;
    // A = Ay Ax
    z.ar = y.ar * x.ar - y.ai * x.ai;
    z.ai = y.ar * x.ai + y.ai * x.ar;
    // B = Ay Bx + By
    z.br = y.ar * x.br - y.ai * x.bi + y.br;
    z.bi = y.ar * x.bi + y.ai * x.br + y.bi;
    // r = max(0, min(rx, (ry - |Bx| cMax) / |Ax|))
    const double rx = std::sqrt(x.r2), ry = std::sqrt(y.r2);
    const double ax = cabs(x.ar, x.ai), bx = cabs(x.br, x.bi);
    double r = ax > 0.0 ? (ry - bx * cMax) / ax : 0.0;
    r = std::max(0.0, std::min(rx, r));
    z.r2 = r * r;
    z.l = x.l + y.l;
    z.pad = 0;
    return z;
}

}  // namespace

void buildBla(const ReferenceOrbit& ref, double cMax, double eps, BlaTable& out) {
    const auto t0 = std::chrono::steady_clock::now();
    out.nodes.clear();
    out.levelOffset.clear();
    out.levelCount.clear();
    out.levels = 0;
    // Single step m uses Z_m and lands on Z_{m+1}; skip m = 0 (Z_0 = 0 is
    // not linearisable) and stop before the last stored entry.
    const int steps = ref.length - 2;
    out.steps = std::max(0, steps);
    if (steps <= 0) return;

    // Level 0.
    out.levelOffset.push_back(0);
    out.levelCount.push_back(steps);
    out.nodes.resize((size_t)steps);
    for (int j = 0; j < steps; ++j) {
        const int m = 1 + j;
        BlaNode& n = out.nodes[(size_t)j];
        n.ar = 2.0 * ref.zr[(size_t)m];
        n.ai = 2.0 * ref.zi[(size_t)m];
        n.br = 1.0;
        n.bi = 0.0;
        // |dz| < eps |2 Z| keeps the dropped dz^2 term below eps relative.
        const double r = eps * cabs(n.ar, n.ai);
        n.r2 = r * r;
        n.l = 1;
        n.pad = 0;
    }
    out.levels = 1;

    // Higher levels: pairwise merge of the level below.
    int prevOff = 0, prevCount = steps;
    while (prevCount > 1) {
        const int count = (prevCount + 1) / 2;
        const int off = (int)out.nodes.size();
        out.nodes.resize((size_t)off + count);
        for (int j = 0; j < count; ++j) {
            const BlaNode& x = out.nodes[(size_t)prevOff + 2 * j];
            if (2 * j + 1 < prevCount) {
                const BlaNode& y = out.nodes[(size_t)prevOff + 2 * j + 1];
                out.nodes[(size_t)off + j] = merge(x, y, cMax);
            } else {
                out.nodes[(size_t)off + j] = x;
            }
        }
        out.levelOffset.push_back(off);
        out.levelCount.push_back(count);
        ++out.levels;
        prevOff = off;
        prevCount = count;
    }
    out.buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

namespace {
// Split a complex double into a float mantissa pair and a shared exponent.
void splitComplex(double r, double i, float& mr, float& mi, int& e) {
    const double mag = std::max(std::fabs(r), std::fabs(i));
    if (!(mag > 0.0) || !std::isfinite(mag)) {
        mr = std::isfinite(mag) ? 0.f : INFINITY;
        mi = 0.f;
        e = 0;
        return;
    }
    int k;
    std::frexp(mag, &k);
    mr = (float)std::ldexp(r, -k);
    mi = (float)std::ldexp(i, -k);
    e = k;
}
}  // namespace

void buildBlaFloat(const BlaTable& in, std::vector<BlaNodeF>& out) {
    out.resize(in.nodes.size());
    for (size_t i = 0; i < in.nodes.size(); ++i) {
        const BlaNode& n = in.nodes[i];
        BlaNodeF f{};
        splitComplex(n.ar, n.ai, f.ar, f.ai, f.ae);
        splitComplex(n.br, n.bi, f.br, f.bi, f.be);
        if (n.r2 > 0.0 && std::isfinite(n.r2) && std::isfinite(n.ar) && std::isfinite(n.ai) &&
            std::isfinite(n.br) && std::isfinite(n.bi)) {
            int k;
            const double m = std::frexp(n.r2, &k);
            f.r2m = (float)m;
            f.r2e = k;
        } else {
            f.r2m = 0.f;
            f.r2e = 0;
        }
        f.l = n.l;
        out[i] = f;
    }
}
