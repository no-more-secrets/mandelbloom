#include "reference.h"
#include <chrono>

void computeReference(const BigFloat& cx, const BigFloat& cy, int maxIter, double bailout,
                      ReferenceOrbit& out) {
    const auto t0 = std::chrono::steady_clock::now();
    const mpfr_prec_t prec = cx.prec();
    mpfr_t zr, zi, zr2, zi2, t;
    mpfr_inits2(prec, zr, zi, zr2, zi2, t, (mpfr_ptr)nullptr);
    mpfr_set_zero(zr, 1);
    mpfr_set_zero(zi, 1);

    out.zr.clear();
    out.zi.clear();
    out.zr.reserve((size_t)maxIter + 1);
    out.zi.reserve((size_t)maxIter + 1);
    out.zr.push_back(0.0);
    out.zi.push_back(0.0);
    out.escaped = false;

    for (int n = 0; n < maxIter; ++n) {
        // zi = 2 zr zi + cy ; zr = zr^2 - zi^2 + cx
        mpfr_sqr(zr2, zr, MPFR_RNDN);
        mpfr_sqr(zi2, zi, MPFR_RNDN);
        mpfr_mul(t, zr, zi, MPFR_RNDN);
        mpfr_mul_2ui(t, t, 1, MPFR_RNDN);
        mpfr_add(zi, t, cy.raw(), MPFR_RNDN);
        mpfr_sub(t, zr2, zi2, MPFR_RNDN);
        mpfr_add(zr, t, cx.raw(), MPFR_RNDN);

        const double dr = mpfr_get_d(zr, MPFR_RNDN);
        const double di = mpfr_get_d(zi, MPFR_RNDN);
        out.zr.push_back(dr);
        out.zi.push_back(di);
        if (dr * dr + di * di > bailout) {
            out.escaped = true;
            break;
        }
    }
    out.length = (int)out.zr.size();
    mpfr_clears(zr, zi, zr2, zi2, t, (mpfr_ptr)nullptr);
    out.computeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
