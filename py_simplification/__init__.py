"""Implements the API"""

import pyvista as pv

from .mesh import _simplify

__all__ = ["collapse_edges"]

def collapse_edges(
    mesh: pv.PolyData,
    target_verts: int | None = None,
    ratio: float = 0.5,
    optimised: bool = True,
    isotropic: bool = False,
    valence_weight: float = 1.0,
    optimal_valence: int = 6,
    use_midpoint: bool = False,
) -> pv.PolyData:
    """_summary_

    Args:
        mesh (pv.PolyData): _description_
        target_verts (int, optional): _description_. Defaults to None.
        ratio (float, optional): _description_. Defaults to 0.5.
        optimised (bool, optional): _description_. Defaults to True.
        isotropic (bool, optional): _description_. Defaults to False.
        valence_weight (float, optional): _description_. Defaults to 1.0.
        optimal_valence (int, optional): _description_. Defaults to 6.
    """
    # validate inputs
    if not isinstance(mesh, pv.PolyData):
        raise TypeError(
            f"Input mesh needs to be type pyvista.PolyData not {type(mesh)}"
        )
    if (target_verts is not None) and (not isinstance(target_verts, int)):
        raise ValueError("target_verts must be of type int if it is not None.")
    if not isinstance(ratio, float) or ((ratio <= 0) or (ratio > 1)):
        raise ValueError("Ratio must be in the range (0,1].")
    if (not isinstance(optimised, bool)) or (not isinstance(isotropic, bool)):
        raise TypeError("optimised and isotropic arguments must be strictly bools.")
    if not isinstance(valence_weight, float) or not isinstance(valence_weight, int):
        raise TypeError("valence_weight must be of type float or int.")
    if not isinstance(optimal_valence, int):
        raise TypeError("optimal_valence must be of type int.")

    # target verts takes precedence
    if target_verts is None:
        target_verts = int(mesh.points.shape[0] * ratio)

    # check for triangle
    if not mesh.is_all_triangles:
        raise AssertionError("Input mesh did not pass triangle faces check.")

    # check for manifoldness
    if not mesh.is_manifold:
        raise ValueError("Input mesh is not manifold.")

    # input verts and faces
    i_verts, i_faces = mesh.points, mesh.regular_faces

    # output verts and faces
    o_verts, o_faces = _simplify(
        i_verts,
        i_faces,
        target_verts=target_verts,
        optimised=optimised,
        isotropic=isotropic,
        valence_weight=valence_weight,
        optimal_valence=optimal_valence,
        use_midpoint=use_midpoint,
    )

    return pv.PolyData().from_regular_faces(o_verts, o_faces)

