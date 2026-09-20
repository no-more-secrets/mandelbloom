#pragma once
#include <vector>
#include "bigfloat.h"

// A reference orbit Z_n = Z_{n-1}^2 + C computed at high precision and
// stored as doubles for the GPU. Z_0 = 0 is entry 0.
struct ReferenceOrbit {
    std::vector<double> zr, zi;
    int length = 0;        // number of valid entries
    bool escaped = false;  // orbit left the bailout radius before maxIter
    double computeMs = 0.0;
};

// Iterate the reference at the precision of cx/cy. Stops at maxIter or when
// |Z|^2 exceeds bailout.
void computeReference(const BigFloat& cx, const BigFloat& cy, int maxIter, double bailout,
                      ReferenceOrbit& out);
