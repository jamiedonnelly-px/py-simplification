from pathlib import Path

import pyvista as pv

from py_simplification import collapse_edges

WORKSPACE = Path(__file__).parents[1]


def test_simplify():
    fpath = WORKSPACE / "data/ankylosaurus.obj"
    input = pv.read(fpath)
    simplified = collapse_edges(input)
    assert simplified.n_points == input.n_points // 2
    assert simplified.is_manifold