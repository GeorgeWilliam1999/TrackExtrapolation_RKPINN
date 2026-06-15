// shims/BackendCommon.h — NVRTC-safe stand-in for Allen's backend/include/BackendCommon.h
//
// Only the UNROLL macro is referenced by RungeKuttaExtrapolator.cuh. We reproduce
// the DEVICE_COMPILER expansion verbatim from Allen's BackendCommon.h
//   #define UNROLL(n) DO_PRAGMA(unroll)
// so the unroll posture matches the production GPU build exactly.
#pragma once

#ifndef DO_PRAGMA
#define DO_PRAGMA(x) _Pragma(#x)
#endif

#ifndef UNROLL
#define UNROLL(n) DO_PRAGMA(unroll)
#endif
