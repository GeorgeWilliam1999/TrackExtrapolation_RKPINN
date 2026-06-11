// cuda_compat.h — Minimal shims so the Allen .cuh extrapolator headers
//                  compile with a standard C++ host compiler (g++/clang++).
#pragma once

#include <cmath>
#include <cstdio>
#include <concepts>

// ── CUDA qualifiers → empty ────────────────────────────────────────────────
#define __device__
#define __host__
#define __global__
#define __inline__

// ── CUDA vector types ──────────────────────────────────────────────────────
struct float2 {
  float x, y;
  friend float2 operator+(const float2& a, const float2& b) { return {a.x + b.x, a.y + b.y}; }
  friend float2 operator-(const float2& a, const float2& b) { return {a.x - b.x, a.y - b.y}; }
  friend float2 operator*(const float2& a, float s)         { return {a.x * s, a.y * s}; }
  friend float2 operator*(float s, const float2& a)         { return {a.x * s, a.y * s}; }
};

struct float3 {
  float x, y, z;
};

inline float2 make_float2(float x, float y)          { return {x, y}; }
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }

// ── CUDA math → standard math ──────────────────────────────────────────────
// sqrtf, fabsf, copysignf are already in global scope via <cmath>
using std::min;
