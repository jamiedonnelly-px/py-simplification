"""Contains the core functionality."""

import copy
import heapq

import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize

LOG_FREQ = 100

def _simplify(
    verts: np.array,
    faces: np.array,
    target_verts: int,
    optimised: bool,
    isotropic: bool,
    valence_weight: float,
    optimal_valence: int,
    use_midpoint: bool,
    verbose: bool = True,
) -> tuple[np.array, np.array]:
    mesh = Mesh(verts, faces, valence_weight, optimal_valence, verbose)

    if isotropic:
        return mesh.edge_based_simplification(
            target_v=target_verts, valence_aware=optimised
        )
    else:
        return mesh.simplification(
            target_v=target_verts, valence_aware=optimised, midpoint=use_midpoint
        )

class Mesh:
    def __init__(
        self,
        verts: np.array,
        faces: np.array,
        valence_weight: float,
        optimal_valence: int,
        verbose: bool
    ):
        self._valence_weight = valence_weight
        self._optimal_valence = optimal_valence
        self._vs, self._faces = verts, faces
        self.verbose = verbose

        self.compute_face_normals()
        if self.verbose:
            print("Computed face normals.")
        
        self.compute_face_center()
        if self.verbose:
            print("Compute face centers.")
        
        self.build_gemm()
        if self.verbose:
            print("Built edge sets.")
        
        self.compute_vert_normals()
        if self.verbose:
            print("Computed vert normals.")

        self.build_v2v()
        if self.verbose:
            print("Built v2v")

        self.build_vf()
        if self.verbose:
            print("Build vf")        

    @property
    def verts(self) -> np.array:
        return self._vs

    @property
    def faces(self) -> np.array:
        return self._faces

    def build_gemm(self):
        self.ve = [[] for _ in self._vs]
        self.vei = [[] for _ in self._vs]
        edge_nb = []
        sides = []
        edge2key = dict()
        edges = []
        edges_count = 0
        nb_count = []
        for face_id, face in enumerate(self._faces):
            faces_edges = []
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                faces_edges.append(cur_edge)
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                faces_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edge_nb.append([-1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1])
                    self.ve[edge[0]].append(edges_count)
                    self.ve[edge[1]].append(edges_count)
                    self.vei[edge[0]].append(0)
                    self.vei[edge[1]].append(1)
                    nb_count.append(0)
                    edges_count += 1
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[
                    faces_edges[(idx + 1) % 3]
                ]
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[
                    faces_edges[(idx + 2) % 3]
                ]
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                sides[edge_key][nb_count[edge_key] - 2] = (
                    nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
                )
                sides[edge_key][nb_count[edge_key] - 1] = (
                    nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
                )
        self.edges = np.array(edges, dtype=np.int32)
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)
        self.sides = np.array(sides, dtype=np.int64)
        self.edges_count = edges_count

    def compute_face_normals(self):
        face_normals = np.cross(
            self._vs[self._faces[:, 1]] - self._vs[self._faces[:, 0]],
            self._vs[self._faces[:, 2]] - self._vs[self._faces[:, 0]],
        )
        norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-24
        face_normals /= norm
        self.fn = face_normals

    def compute_vert_normals(self):
        vert_normals = np.zeros((3, len(self._vs)))
        face_normals = self.fn
        faces = self._faces

        nv = len(self._vs)
        nf = len(faces)
        mat_rows = faces.reshape(-1)
        mat_cols = np.array([[i] * 3 for i in range(nf)]).reshape(-1)
        mat_vals = np.ones(len(mat_rows))
        f2v_mat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(nv, nf))
        vert_normals = sp.sparse.csr_matrix.dot(f2v_mat, face_normals)
        vert_normals = normalize(vert_normals, norm="l2", axis=1)
        self.vn = vert_normals

    def compute_face_center(self):
        self.fc = np.sum(self._vs[self._faces], 1) / 3.0

    def build_uni_lap(self):
        """compute uniform laplacian matrix"""
        edges = self.edges
        ve = self.ve

        sub_mesh_vv = [edges[v_e, :].reshape(-1) for v_e in ve]
        sub_mesh_vv = [
            set(vv.tolist()).difference(set([i])) for i, vv in enumerate(sub_mesh_vv)
        ]

        num_verts = self._vs.shape[0]
        mat_rows = [
            np.array([i] * len(vv), dtype=np.int64) for i, vv in enumerate(sub_mesh_vv)
        ]
        mat_rows = np.concatenate(mat_rows)
        mat_cols = [np.array(list(vv), dtype=np.int64) for vv in sub_mesh_vv]
        mat_cols = np.concatenate(mat_cols)
        mat_vals = np.ones_like(mat_rows, dtype=np.float32) * -1.0
        neig_mat = sp.sparse.csr_matrix(
            (mat_vals, (mat_rows, mat_cols)), shape=(num_verts, num_verts)
        )
        sum_count = sp.sparse.csr_matrix.dot(
            neig_mat, np.ones((num_verts, 1), dtype=np.float32)
        )

        mat_rows_ident = np.array([i for i in range(num_verts)])
        mat_cols_ident = np.array([i for i in range(num_verts)])
        mat_ident = np.array([-s for s in sum_count[:, 0]])

        mat_rows = np.concatenate([mat_rows, mat_rows_ident], axis=0)
        mat_cols = np.concatenate([mat_cols, mat_cols_ident], axis=0)
        mat_vals = np.concatenate([mat_vals, mat_ident], axis=0)

        self.lapmat = sp.sparse.csr_matrix(
            (mat_vals, (mat_rows, mat_cols)), shape=(num_verts, num_verts)
        )

    def build_vf(self):
        vf = [set() for _ in range(len(self._vs))]
        for i, f in enumerate(self._faces):
            vf[f[0]].add(i)
            vf[f[1]].add(i)
            vf[f[2]].add(i)
        self.vf = vf

    def build_v2v(self):
        v2v = [[] for _ in range(len(self._vs))]
        for i, e in enumerate(self.edges):
            v2v[e[0]].append(e[1])
            v2v[e[1]].append(e[0])
        self.v2v = v2v

    def simplification(self, target_v: int, valence_aware: bool, midpoint: bool) -> tuple[np.array]:
        vs, vf, fn, fc, edges = self._vs, self.vf, self.fn, self.fc, self.edges

        """ 1. compute Q for each vertex """
        Q_s = [[] for _ in range(len(vs))]
        E_s = [[] for _ in range(len(vs))]
        for i, v in enumerate(vs):
            f_s = np.array(list(vf[i]))
            fc_s = fc[f_s]
            fn_s = fn[f_s]
            d_s = -1.0 * np.sum(fn_s * fc_s, axis=1, keepdims=True)
            abcd_s = np.concatenate([fn_s, d_s], axis=1)
            Q_s[i] = np.matmul(abcd_s.T, abcd_s)
            v4 = np.concatenate([v, np.array([1])])
            E_s[i] = np.matmul(v4, np.matmul(Q_s[i], v4.T))

        """ 2. compute E for every possible pairs and create heapq """
        E_heap = []
        for i, e in enumerate(edges):
            v_0, v_1 = vs[e[0]], vs[e[1]]
            Q_0, Q_1 = Q_s[e[0]], Q_s[e[1]]
            Q_new = Q_0 + Q_1

            if midpoint:
                v_new = 0.5 * (v_0 + v_1)
                v4_new = np.concatenate([v_new, np.array([1])])
            else:
                Q_lp = np.eye(4)
                Q_lp[:3] = Q_new[:3]
                try:
                    Q_lp_inv = np.linalg.inv(Q_lp)
                    v4_new = np.matmul(
                        Q_lp_inv, np.array([[0, 0, 0, 1]]).reshape(-1, 1)
                    ).reshape(-1)
                except Exception:
                    v_new = 0.5 * (v_0 + v_1)
                    v4_new = np.concatenate([v_new, np.array([1])])

            valence_penalty = 1
            if valence_aware:
                merged_faces = vf[e[0]].intersection(vf[e[1]])
                valence_new = len(vf[e[0]].union(vf[e[1]]).difference(merged_faces))
                valence_penalty = self.valence_weight(valence_new)

            E_new = np.matmul(v4_new, np.matmul(Q_new, v4_new.T)) * valence_penalty
            heapq.heappush(E_heap, (E_new, (e[0], e[1])))

        """ 3. collapse minimum-error vertex """
        simp_mesh = copy.deepcopy(self)

        vi_mask = np.ones([len(simp_mesh._vs)]).astype(np.bool_)
        fi_mask = np.ones([len(simp_mesh._faces)]).astype(np.bool_)

        vert_map = [{i} for i in range(len(simp_mesh._vs))]
        step_counter = 0
        total = np.sum(vi_mask) - target_v
        while np.sum(vi_mask) > target_v:
            if len(E_heap) == 0:
                print("[Warning]: edge cannot be collapsed anymore!")
                break

            E_0, (vi_0, vi_1) = heapq.heappop(E_heap)

            if (not vi_mask[vi_0]) or (not vi_mask[vi_1]):
                continue

            """ edge collapse """
            shared_vv = list(
                set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1]))
            )
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])

            if len(shared_vv) != 2:
                """ non-manifold! """
                continue

            elif len(merged_faces) != 2:
                """ boundary """
                continue

            else:
                self.edge_collapse(
                    simp_mesh,
                    vi_0,
                    vi_1,
                    merged_faces,
                    vi_mask,
                    fi_mask,
                    vert_map,
                    Q_s,
                    E_heap,
                    valence_aware=valence_aware,
                )
                if ((step_counter+1) % LOG_FREQ == 0) and self.verbose:
                    print(f"{step_counter}/{total} edges collapsed...{(step_counter/total)*100:.0f}% done.")
                step_counter += 1

        verts, faces = self.extract_arrays(simp_mesh, vi_mask, fi_mask, vert_map)

        return verts, faces

    def edge_based_simplification(self, target_v: int, valence_aware=True) -> tuple[np.array]:
        vs, edges = self._vs, self.edges
        edge_len = vs[edges][:, 0, :] - vs[edges][:, 1, :]
        edge_len = np.linalg.norm(edge_len, axis=1)
        edge_len_heap = np.stack([edge_len, np.arange(len(edge_len))], axis=1).tolist()
        heapq.heapify(edge_len_heap)

        """ 2. compute E for every possible pairs and create heapq """
        E_heap = []
        for i, e in enumerate(edges):
            heapq.heappush(E_heap, (edge_len[i], (e[0], e[1])))

        """ 3. collapse minimum-error vertex """
        simp_mesh = copy.deepcopy(self)

        vi_mask = np.ones([len(simp_mesh._vs)]).astype(np.bool_)
        fi_mask = np.ones([len(simp_mesh._faces)]).astype(np.bool_)

        vert_map = [{i} for i in range(len(simp_mesh._vs))]
        step_counter = 0
        total = np.sum(vi_mask) - target_v
        while np.sum(vi_mask) > target_v:
            if len(E_heap) == 0:
                print("[Warning]: edge cannot be collapsed anymore!")
                break

            E_0, (vi_0, vi_1) = heapq.heappop(E_heap)

            if (not vi_mask[vi_0]) or (not vi_mask[vi_1]):
                continue

            """ edge collapse """
            shared_vv = list(
                set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1]))
            )
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])

            if len(shared_vv) != 2:
                """ non-manifold! """
                # print("non-manifold can be occured!!" , len(shared_vv))
                continue

            elif len(merged_faces) != 2:
                """ boundary """
                # print("boundary edge cannot be collapsed!")
                continue

            else:
                self.edge_based_collapse(
                    simp_mesh,
                    vi_0,
                    vi_1,
                    merged_faces,
                    vi_mask,
                    fi_mask,
                    vert_map,
                    E_heap,
                    valence_aware=valence_aware,
                )
                if ((step_counter+1) % LOG_FREQ == 0) and self.verbose:
                    print(f"{step_counter}/{total} edges collapsed...{(step_counter/total)*100:.0f}% done.")
                step_counter += 1

        verts, faces = Mesh.extract_arrays(simp_mesh, vi_mask, fi_mask, vert_map)

        return verts, faces

    def edge_collapse(
        self,
        simp_mesh,
        vi_0,
        vi_1,
        merged_faces,
        vi_mask,
        fi_mask,
        vert_map,
        Q_s,
        E_heap,
        valence_aware,
    ):
        shared_vv = list(
            set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1]))
        )
        new_vi_0 = (
            set(simp_mesh.v2v[vi_0])
            .union(set(simp_mesh.v2v[vi_1]))
            .difference({vi_0, vi_1})
        )
        simp_mesh.vf[vi_0] = (
            simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)
        )
        simp_mesh.vf[vi_1] = set()
        simp_mesh.vf[shared_vv[0]] = simp_mesh.vf[shared_vv[0]].difference(merged_faces)
        simp_mesh.vf[shared_vv[1]] = simp_mesh.vf[shared_vv[1]].difference(merged_faces)

        simp_mesh.v2v[vi_0] = list(new_vi_0)
        for v in simp_mesh.v2v[vi_1]:
            if v != vi_0:
                simp_mesh.v2v[v] = list(
                    set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0})
                )
        simp_mesh.v2v[vi_1] = []
        vi_mask[vi_1] = False

        vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])
        vert_map[vi_0] = vert_map[vi_0].union({vi_1})
        vert_map[vi_1] = set()

        fi_mask[np.array(list(merged_faces)).astype(np.int32)] = False

        simp_mesh._vs[vi_0] = 0.5 * (simp_mesh._vs[vi_0] + simp_mesh._vs[vi_1])

        """ recompute E """
        Q_0 = Q_s[vi_0]
        for vv_i in simp_mesh.v2v[vi_0]:
            v_mid = 0.5 * (simp_mesh._vs[vi_0] + simp_mesh._vs[vv_i])
            Q_1 = Q_s[vv_i]
            Q_new = Q_0 + Q_1
            v4_mid = np.concatenate([v_mid, np.array([1])])

            valence_penalty = 1
            if valence_aware:
                merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vv_i])
                valence_new = len(
                    simp_mesh.vf[vi_0]
                    .union(simp_mesh.vf[vv_i])
                    .difference(merged_faces)
                )
                valence_penalty = self.valence_weight(valence_new)

            E_new = np.matmul(v4_mid, np.matmul(Q_new, v4_mid.T)) * valence_penalty
            heapq.heappush(E_heap, (E_new, (vi_0, vv_i)))

    def edge_based_collapse(
        self,
        simp_mesh,
        vi_0,
        vi_1,
        merged_faces,
        vi_mask,
        fi_mask,
        vert_map,
        E_heap,
        valence_aware,
    ):
        shared_vv = list(
            set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1]))
        )
        new_vi_0 = (
            set(simp_mesh.v2v[vi_0])
            .union(set(simp_mesh.v2v[vi_1]))
            .difference({vi_0, vi_1})
        )
        simp_mesh.vf[vi_0] = (
            simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)
        )
        simp_mesh.vf[vi_1] = set()
        simp_mesh.vf[shared_vv[0]] = simp_mesh.vf[shared_vv[0]].difference(merged_faces)
        simp_mesh.vf[shared_vv[1]] = simp_mesh.vf[shared_vv[1]].difference(merged_faces)

        simp_mesh.v2v[vi_0] = list(new_vi_0)
        for v in simp_mesh.v2v[vi_1]:
            if v != vi_0:
                simp_mesh.v2v[v] = list(
                    set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0})
                )
        simp_mesh.v2v[vi_1] = []
        vi_mask[vi_1] = False

        vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])
        vert_map[vi_0] = vert_map[vi_0].union({vi_1})
        vert_map[vi_1] = set()

        fi_mask[np.array(list(merged_faces)).astype(np.int32)] = False

        simp_mesh._vs[vi_0] = 0.5 * (simp_mesh._vs[vi_0] + simp_mesh._vs[vi_1])

        """ recompute E """
        for vv_i in simp_mesh.v2v[vi_0]:
            edge_len = np.linalg.norm(simp_mesh._vs[vi_0] - simp_mesh._vs[vv_i])
            valence_penalty = 1
            if valence_aware:
                merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vv_i])
                valence_new = len(
                    simp_mesh.vf[vi_0]
                    .union(simp_mesh.vf[vv_i])
                    .difference(merged_faces)
                )
                valence_penalty = self.valence_weight(valence_new)
                edge_len *= valence_penalty

            heapq.heappush(E_heap, (edge_len, (vi_0, vv_i)))

    def valence_weight(self, valence_new):
        valence_penalty = (
            abs(valence_new - self._optimal_valence) * self._valence_weight + 1
        )
        if valence_new == 3:
            valence_penalty *= 100000
        return valence_penalty

    @staticmethod
    def extract_arrays(simp_mesh, vi_mask, fi_mask, vert_map):
        face_map = dict(zip(np.arange(len(vi_mask)), np.cumsum(vi_mask) - 1))
        simp_mesh._vs = simp_mesh._vs[vi_mask]

        vert_dict = {}
        for i, vm in enumerate(vert_map):
            for j in vm:
                vert_dict[j] = i

        for i, f in enumerate(simp_mesh._faces):
            for j in range(3):
                if f[j] in vert_dict:
                    simp_mesh._faces[i][j] = vert_dict[f[j]]

        simp_mesh._faces = simp_mesh._faces[fi_mask]
        for i, f in enumerate(simp_mesh._faces):
            for j in range(3):
                simp_mesh._faces[i][j] = face_map[f[j]]

        return simp_mesh.verts, simp_mesh.faces