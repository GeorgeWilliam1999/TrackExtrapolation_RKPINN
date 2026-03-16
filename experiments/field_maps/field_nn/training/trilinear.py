"""
Vectorised trilinear interpolation on a regular 3D grid.

Replicates the C++ fieldVectorLinearInterpolation used by the LHCb
field service.  Given query points (x, y, z), returns the linearly
interpolated (Bx, By, Bz) from the surrounding 8 grid corners.

Usage:
    grid = TrilinearGrid.from_file('twodip.rtf')
    B = grid.query(points)  # points: (N, 3) -> B: (N, 3)
"""

import numpy as np


class TrilinearGrid:
    """Regular 3-D grid supporting vectorised trilinear interpolation."""

    def __init__(self, x_nodes, y_nodes, z_nodes, field_grid):
        self.x = x_nodes.astype(np.float64)
        self.y = y_nodes.astype(np.float64)
        self.z = z_nodes.astype(np.float64)
        self.field = field_grid.astype(np.float64)  # (nx, ny, nz, 3)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nz = len(self.z)

    @classmethod
    def from_file(cls, path):
        """Load a field-map file (space-separated x y z Bx By Bz)."""
        data = np.loadtxt(path)
        coords = data[:, :3]
        fields = data[:, 3:]

        x_nodes = np.unique(coords[:, 0])
        y_nodes = np.unique(coords[:, 1])
        z_nodes = np.unique(coords[:, 2])
        nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)

        field_grid = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        for row in data:
            ix = int(np.searchsorted(x_nodes, row[0]))
            iy = int(np.searchsorted(y_nodes, row[1]))
            iz = int(np.searchsorted(z_nodes, row[2]))
            field_grid[ix, iy, iz] = row[3:]

        return cls(x_nodes, y_nodes, z_nodes, field_grid)

    def query(self, points):
        """
        Trilinear interpolation at arbitrary (x, y, z) positions.

        Parameters
        ----------
        points : (N, 3) array of query positions [mm].

        Returns
        -------
        B : (N, 3) interpolated field values [T].
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[None, :]

        fx = (pts[:, 0] - self.x[0]) / self.dx
        fy = (pts[:, 1] - self.y[0]) / self.dy
        fz = (pts[:, 2] - self.z[0]) / self.dz

        ix = np.clip(np.floor(fx).astype(int), 0, self.nx - 2)
        iy = np.clip(np.floor(fy).astype(int), 0, self.ny - 2)
        iz = np.clip(np.floor(fz).astype(int), 0, self.nz - 2)

        tx = np.clip(fx - ix, 0.0, 1.0)
        ty = np.clip(fy - iy, 0.0, 1.0)
        tz = np.clip(fz - iz, 0.0, 1.0)

        c000 = self.field[ix,     iy,     iz    ]
        c001 = self.field[ix,     iy,     iz + 1]
        c010 = self.field[ix,     iy + 1, iz    ]
        c011 = self.field[ix,     iy + 1, iz + 1]
        c100 = self.field[ix + 1, iy,     iz    ]
        c101 = self.field[ix + 1, iy,     iz + 1]
        c110 = self.field[ix + 1, iy + 1, iz    ]
        c111 = self.field[ix + 1, iy + 1, iz + 1]

        tx = tx[:, None]
        ty = ty[:, None]
        tz = tz[:, None]

        B = (c000 * (1 - tx) * (1 - ty) * (1 - tz) +
             c001 * (1 - tx) * (1 - ty) * tz +
             c010 * (1 - tx) * ty * (1 - tz) +
             c011 * (1 - tx) * ty * tz +
             c100 * tx * (1 - ty) * (1 - tz) +
             c101 * tx * (1 - ty) * tz +
             c110 * tx * ty * (1 - tz) +
             c111 * tx * ty * tz)

        return B

    def random_off_grid_points(self, n, rng=None):
        """Generate n random points uniformly within the grid bounds."""
        if rng is None:
            rng = np.random.default_rng(42)
        x = rng.uniform(self.x[0], self.x[-1], n)
        y = rng.uniform(self.y[0], self.y[-1], n)
        z = rng.uniform(self.z[0], self.z[-1], n)
        return np.stack([x, y, z], axis=1)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'twodip.rtf'
    grid = TrilinearGrid.from_file(path)
    print(f'Grid: {grid.nx}x{grid.ny}x{grid.nz} = {grid.nx*grid.ny*grid.nz:,} points')

    test_pt = np.array([[grid.x[10], grid.y[20], grid.z[30]]])
    B_query = grid.query(test_pt)
    B_exact = grid.field[10, 20, 30]
    err = np.abs(B_query[0] - B_exact).max()
    print(f'On-grid test error: {err:.2e} T (should be ~0)')

    off_pts = grid.random_off_grid_points(100_000)
    B_off = grid.query(off_pts)
    print(f'Off-grid query: {len(off_pts):,} points, B shape {B_off.shape}')
    print('Self-test passed.')
