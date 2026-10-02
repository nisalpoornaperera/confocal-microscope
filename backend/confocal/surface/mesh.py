"""Delaunay triangulation of the used points and the triangle mesh built from it.

One triangulation serves three purposes in a reconstruction: linear and
cubic interpolation, the confidence map and the 3-D mesh. Building it once
matters on the Pi: for a 100 x 100 scan the triangulation and its
barycentric transform take most of the reconstruction time.

Triangulation runs on coordinates normalised to the scan step (Qhull is
best conditioned for O(1) coordinates); vertex indices are unaffected.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError

from confocal.models.surface import MeshData

#: Relative singular-value threshold below which a point set counts as collinear.
_COLLINEAR_TOLERANCE = 1e-9


def is_two_dimensional(xy: NDArray[np.float64]) -> bool:
    """True when at least three points are not all on one line."""
    if xy.shape[0] < 3:
        return False
    centred = xy - xy.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    return bool(singular[0] > 0.0 and singular[-1] > _COLLINEAR_TOLERANCE * singular[0])


def triangulate(xy: NDArray[np.float64]) -> Delaunay | None:
    """Delaunay triangulation, or None for fewer than 3 non-collinear points."""
    if not is_two_dimensional(xy):
        return None
    try:
        return Delaunay(xy)
    except QhullError:
        return None


def build_mesh(
    triangulation: Delaunay,
    x_um: NDArray[np.float64],
    y_um: NDArray[np.float64],
    z_um: NDArray[np.float64],
) -> MeshData:
    """Triangle mesh with consistently counter-clockwise faces (seen from +Z).

    Consistent winding gives a renderer outward (+Z) normals for every face;
    degenerate (zero-area) triangles are dropped.
    """
    faces = np.asarray(triangulation.simplices, dtype=np.int64)
    x0, y0 = x_um[faces[:, 0]], y_um[faces[:, 0]]
    cross = (x_um[faces[:, 1]] - x0) * (y_um[faces[:, 2]] - y0) - (x_um[faces[:, 2]] - x0) * (
        y_um[faces[:, 1]] - y0
    )
    clockwise = cross < 0.0
    faces[clockwise] = faces[clockwise][:, [0, 2, 1]]
    faces = faces[np.abs(cross) > 0.0]
    vertices = [(float(x), float(y), float(z)) for x, y, z in zip(x_um, y_um, z_um, strict=True)]
    return MeshData(
        vertices=vertices,
        faces=[(int(a), int(b), int(c)) for a, b, c in faces],
    )
