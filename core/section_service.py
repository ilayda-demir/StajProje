# core/section_service.py
import numpy as np
import trimesh

class SectionService:
    def __init__(self):
        pass

    @staticmethod
    def compute_section(mesh: trimesh.Trimesh, point: np.ndarray, axis: str):
        """
        Shapely/GEOS KULLANMADAN:
        - trimesh.intersections.mesh_plane -> 3B segmentler
        - Segment uçlarını toleransla eşleyip polylinelere stitch et
        - NaN/tek-nokta filtreleri
        Döner: [np.ndarray(K,3), ...]
        """
        if mesh is None or point is None:
            return []

        axis = (axis or "z").lower()
        if axis == "x":
            n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif axis == "y":
            n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int32).reshape(-1, 3)
        if V.size == 0 or F.size == 0:
            return []

        # 1) Düzlem-mesh kesişim segmentleri (saf, stabil)
        try:
            segs = trimesh.intersections.mesh_plane(
                mesh=mesh,
                plane_normal=n,
                plane_origin=np.asarray(point, dtype=np.float64),
            )
        except Exception:
            return []
        if segs is None or len(segs) == 0:
            return []

        segs = np.asarray(segs, dtype=np.float64)  # (N,2,3)

        # 2) Uçları toleransla eşleyip polylinelere birleştir
        EPS = 1e-6
        def key(p):  # toleranslı hash
            return tuple(np.round(p / EPS).astype(np.int64).tolist())

        from collections import defaultdict
        adj = defaultdict(list)
        for i, (a, b) in enumerate(segs):
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                continue
            adj[key(a)].append((i, 0))
            adj[key(b)].append((i, 1))

        unused = set(range(len(segs)))
        paths = []

        while unused:
            i0 = unused.pop()
            a, b = segs[i0]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            path = [a, b]

            for forward in (True, False):
                cur = path[-1] if forward else path[0]
                while True:
                    lst = adj.get(key(cur), [])
                    nxt = None
                    for si, end_id in lst:
                        if si in unused:
                            pa, pb = segs[si]
                            other = pb if end_id == 0 else pa
                            if np.all(np.isfinite(other)):
                                nxt = (si, other)
                                break
                    if nxt is None:
                        break
                    si, other = nxt
                    unused.remove(si)
                    if forward:
                        # ardışık ayni nokta olmasın
                        if not np.allclose(path[-1], other, atol=EPS):
                            path.append(other)
                        cur = other
                    else:
                        if not np.allclose(path[0], other, atol=EPS):
                            path.insert(0, other)
                        cur = other

            P = np.asarray(path, dtype=np.float64)
            # NaN/tek-nokta/gereksiz tekrarları at
            if P.shape[0] >= 2 and np.all(np.isfinite(P)):
                # art arda aynı noktaları sıkıştır
                keep = [0]
                for i in range(1, P.shape[0]):
                    if not np.allclose(P[i], P[keep[-1]], atol=EPS):
                        keep.append(i)
                if len(keep) >= 2:
                    paths.append(P[keep])

        return paths
