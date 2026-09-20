#pragma once
// Thin RAII wrapper over MPFR for the handful of operations the viewer
// needs: view center arithmetic and the reference orbit.
#include <mpfr.h>
#include <string>
#include <utility>

class BigFloat {
public:
    explicit BigFloat(mpfr_prec_t prec = 64) { mpfr_init2(v_, prec); mpfr_set_zero(v_, 1); }
    BigFloat(const BigFloat& o) { mpfr_init2(v_, mpfr_get_prec(o.v_)); mpfr_set(v_, o.v_, MPFR_RNDN); }
    BigFloat(BigFloat&& o) noexcept { mpfr_init2(v_, 64); mpfr_swap(v_, o.v_); }
    BigFloat& operator=(const BigFloat& o) {
        if (this != &o) { mpfr_set_prec(v_, mpfr_get_prec(o.v_)); mpfr_set(v_, o.v_, MPFR_RNDN); }
        return *this;
    }
    BigFloat& operator=(BigFloat&& o) noexcept { mpfr_swap(v_, o.v_); return *this; }
    ~BigFloat() { mpfr_clear(v_); }

    mpfr_prec_t prec() const { return mpfr_get_prec(v_); }
    // Change precision, keeping the value (rounded if shrinking).
    void setPrec(mpfr_prec_t p) { mpfr_prec_round(v_, p, MPFR_RNDN); }

    void set(double d) { mpfr_set_d(v_, d, MPFR_RNDN); }
    bool set(const std::string& s) { return mpfr_set_str(v_, s.c_str(), 10, MPFR_RNDN) == 0; }
    double toDouble() const { return mpfr_get_d(v_, MPFR_RNDN); }
    std::string toString(int digits) const {
        char* s = nullptr;
        int n = mpfr_asprintf(&s, "%.*Rg", digits, v_);
        std::string r = n >= 0 && s ? s : "?";
        if (s) mpfr_free_str(s);
        return r;
    }

    void addDouble(double d) { mpfr_add_d(v_, v_, d, MPFR_RNDN); }
    void subDouble(double d) { mpfr_sub_d(v_, v_, d, MPFR_RNDN); }

    mpfr_ptr raw() { return v_; }
    mpfr_srcptr raw() const { return v_; }

private:
    mpfr_t v_;
};
