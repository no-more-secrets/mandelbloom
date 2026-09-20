#pragma once
#include <vector>
#include "reference.h"

// One bilinear approximation node: over l reference steps starting at
// index m,  dz_{n+l} ~= A dz_n + B dc,  valid while |dz_n|^2 < r2 (and
// |dc| <= cMax, which held when the table was built).
struct BlaNode {
    double ar, ai;  // A
    double br, bi;  // B
    double r2;      // validity radius squared
    int l;          // steps covered
    int pad;
};

// Table layout: level k holds nodes covering 2^k steps (fewer at the end),
// node j of level k starts at reference index m = 1 + j * 2^k.
struct BlaTable {
    std::vector<BlaNode> nodes;
    std::vector<int> levelOffset;  // start of each level in nodes
    std::vector<int> levelCount;
    int levels = 0;
    int steps = 0;  // single steps available: M - 2
    double buildMs = 0.0;
};

// cMax: largest |dc| in the image. eps: relative truncation tolerance for
// dropping the dz^2 term (2^-24 is float-level accuracy).
void buildBla(const ReferenceOrbit& ref, double cMax, double eps, BlaTable& out);

// Float form of a node: mantissas with power-of-two exponents so deep
// zooms fit in float. A = (ar, ai) * 2^ae, B = (br, bi) * 2^be,
// r^2 = r2m * 2^r2e (r2m == 0 means never valid). 40 bytes.
struct BlaNodeF {
    float ar, ai;
    float br, bi;
    int ae, be;
    float r2m;
    int r2e;
    int l;
    int pad;
};

// Convert a double table to float nodes (same layout and offsets).
void buildBlaFloat(const BlaTable& in, std::vector<BlaNodeF>& out);

