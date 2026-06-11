// MagneticField.cuh — Standalone mock of the LHCb magnetic field.
//
// The real Allen MagneticField::Magfield uses either CUDA textures or a
// 3-D float array with trilinear interpolation from a measured field map.
// Here we model the LHCb dipole analytically so the extrapolators can
// be exercised without loading binary field-map data.
//
// The model: By peaks at ~ -1.1 T near z = 5200 mm (between UT and SciFi),
// with a Gaussian falloff. Small Bx and Bz fringe components are included.
//
// UNITS: In the Gaudi/Geant4 internal system (mm, ns, MeV, eplus),
//   1 Tesla = 1e-3.  The peak LHCb dipole field is ~1.1 T = 1.1e-3.
#pragma once
#include <cmath>
#include "cuda_compat.h"

namespace MagneticField {
  struct Magfield {
    // Analytical LHCb-like dipole field for standalone tests.
    // Field values are in Gaudi internal units (1 Tesla = 1e-3).
    float3 fieldVectorLinearInterpolation(float3 pos) const
    {
      float By = -1.1e-3f * std::exp(-0.5f * ((pos.z - 5200.0f) / 1800.0f)
                                             * ((pos.z - 5200.0f) / 1800.0f));
      float Bx = 0.02f * By * (pos.y / 1000.0f);
      float Bz = 0.01f * By * (pos.x / 1000.0f);
      return {Bx, By, Bz};
    }

    // Unused in standalone mode, but present in the real struct.
    float invDx{1}, invDy{1}, invDz{1};
    float minX{0}, minY{0}, minZ{0};
  };
} // namespace MagneticField
