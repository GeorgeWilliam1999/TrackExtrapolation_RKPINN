#!/usr/bin/env python3
"""P0.0b — loader + validation for the canonical LHCb field map.

Canonical map: /cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.down.bin
(the file Allen consumes as magfield.bin; format per
 Allen/integration/non_event_data/src/MagneticField.cpp:
 4f invDxyz | 4i Nxyz | 4f minXYZ | N*4 floats (Bx,By,Bz,pad), x-fastest).

Provides Vec-trilinear interpolation matching Allen's
fieldVectorLinearInterpolation, with auto unit detection (Tesla vs Gaudi).
"""
from __future__ import annotations

import numpy as np

V8R1_DOWN = "/cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.down.bin"


class FieldV8R1:
    def __init__(self, path: str = V8R1_DOWN):
        raw = np.fromfile(path, dtype=np.float32)
        invD = raw[0:3]
        N = raw[4:8].view(np.int32)[0:3]
        mn = raw[8:11]
        B = raw[12:12 + int(N[0]) * int(N[1]) * int(N[2]) * 4].reshape(-1, 4)
        self.invD = invD.astype(np.float64)
        self.N = N.astype(np.int64)
        self.min = mn.astype(np.float64)
        nx, ny, nz = (int(N[0]), int(N[1]), int(N[2]))
        # x-fastest ordering: index = ix + Nx*(iy + Ny*iz)
        self.Bx = B[:, 0].reshape(nz, ny, nx).astype(np.float64)
        self.By = B[:, 1].reshape(nz, ny, nx).astype(np.float64)
        self.Bz = B[:, 2].reshape(nz, ny, nx).astype(np.float64)
        # unit detection: peak |By| ~1 => Tesla; ~1e-3 => Gaudi units (MeV ns / mm^2)
        peak = float(np.nanmax(np.abs(self.By)))
        if 0.1 < peak < 10:
            self.scale = 1.0
        elif 1e-4 < peak < 1e-2:
            self.scale = 1.0 / 1e-3  # Gaudi: 1 T = 1e-3
        else:
            raise ValueError(f"unrecognised field units, peak|By|={peak}")
        self._peak = peak

    def info(self) -> str:
        d = 1.0 / self.invD
        mx = self.min + (self.N - 1) * d
        return (f"N={tuple(self.N)} d={tuple(np.round(d,1))} "
                f"min={tuple(np.round(self.min,1))} max={tuple(np.round(mx,1))} "
                f"peak|By|(raw)={self._peak:.5g} scale->T={self.scale:g}")

    def __call__(self, x, y, z):
        """Vectorised trilinear, returns (Bx,By,Bz) in Tesla."""
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        fx = (x - self.min[0]) * self.invD[0]
        fy = (y - self.min[1]) * self.invD[1]
        fz = (z - self.min[2]) * self.invD[2]
        ix = np.clip(fx.astype(np.int64), 0, self.N[0] - 2)
        iy = np.clip(fy.astype(np.int64), 0, self.N[1] - 2)
        iz = np.clip(fz.astype(np.int64), 0, self.N[2] - 2)
        tx = np.clip(fx - ix, 0.0, 1.0); ty = np.clip(fy - iy, 0.0, 1.0)
        tz = np.clip(fz - iz, 0.0, 1.0)

        def tri(G):
            c000 = G[iz, iy, ix];     c100 = G[iz, iy, ix + 1]
            c010 = G[iz, iy + 1, ix]; c110 = G[iz, iy + 1, ix + 1]
            c001 = G[iz + 1, iy, ix]; c101 = G[iz + 1, iy, ix + 1]
            c011 = G[iz + 1, iy + 1, ix]; c111 = G[iz + 1, iy + 1, ix + 1]
            c00 = c000 * (1 - tx) + c100 * tx
            c10 = c010 * (1 - tx) + c110 * tx
            c01 = c001 * (1 - tx) + c101 * tx
            c11 = c011 * (1 - tx) + c111 * tx
            c0 = c00 * (1 - ty) + c10 * ty
            c1 = c01 * (1 - ty) + c11 * ty
            return (c0 * (1 - tz) + c1 * tz) * self.scale

        return tri(self.Bx), tri(self.By), tri(self.Bz)


if __name__ == "__main__":
    f = FieldV8R1()
    print("v8r1.down:", f.info())
    print(f"By(0,0,5000) = {f(0,0,5000)[1]:+.4f} T")
    zs = np.linspace(2665.0, 7826.0, 2065)
    by = f(np.zeros_like(zs), np.zeros_like(zs), zs)[1]
    print(f"on-axis int By dz (2665->7826) = {np.trapezoid(by, zs):+.1f} T*mm")
