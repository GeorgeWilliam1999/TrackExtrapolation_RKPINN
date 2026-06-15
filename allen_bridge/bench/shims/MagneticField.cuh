// shims/MagneticField.cuh — NVRTC-safe MagneticField::Magfield.
//
// Allen's real device/event_model/common/include/MagneticField.cuh declares the
// texture-backed struct with cudaArray_t members (host handles NVRTC does not
// know). We therefore provide a trimmed struct that keeps ONLY the device-visible
// members (the three texture-object handles + the affine grid params) and copies
// the texture-path fieldVectorLinearInterpolation() body BYTE-FOR-BYTE from Allen
// (the MAGFIELD_USE_TEXTURE + __CUDA_ARCH__ branch). The production GPU build uses
// exactly this path (CMake: option MAGFIELD_USE_TEXTURE default ON for CUDA).
//
// The device-executed instructions (coordinate transform + 3x tex3D<float>) are
// identical to production, so the memory-access cost we time is faithful. Only the
// host-side array handles, irrelevant to the kernel, are omitted.
#pragma once

namespace MagneticField {
  struct Magfield {
    // --- verbatim from Allen (MAGFIELD_USE_TEXTURE && __CUDA_ARCH__ branch) ---
    __device__ float3 fieldVectorLinearInterpolation(float3 pos) const
    {
      const float x = (pos.x - minX) * invDx;
      const float y = (pos.y - minY) * invDy;
      const float z = (pos.z - minZ) * invDz;
      return {tex3D<float>(tex_Bx, x, y, z), tex3D<float>(tex_By, x, y, z), tex3D<float>(tex_Bz, x, y, z)};
    }

    cudaTextureObject_t tex_Bx;
    cudaTextureObject_t tex_By;
    cudaTextureObject_t tex_Bz;

    float invDx, invDy, invDz;
    float minX, minY, minZ;
  };
} // namespace MagneticField
