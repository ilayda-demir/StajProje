# core/section_service.py
import numpy as np
import trimesh


class SectionService:
    def __init__(self):
        pass

    # ----- Yardımcı: Geometriyi (V,F) olarak ver -----
    @staticmethod
    def _as_VF(geom):
        """
        geom: trimesh.Trimesh veya (V, F) tuple'ı olabilir.
        V -> (N,3) float64, F -> (M,3) int32
        """
        if isinstance(geom, trimesh.Trimesh):
            V = np.asarray(geom.vertices, dtype=np.float64)
            F = np.asarray(geom.faces,    dtype=np.int32).reshape(-1, 3)
        else:
            V, F = geom
            V = np.asarray(V, dtype=np.float64)
            F = np.asarray(F, dtype=np.int32).reshape(-1, 3)
        return V, F

    # ----- PROJEKSİYON TABANLI PICK -----
    @staticmethod
    def pick_point_projective(geom, MV, P, vp, mouse_win_x, mouse_win_y, eps_draw: float = 3e-3):
        """
        Ekrana projeksiyon tabanlı picking:
        - V/F'yi MVP ile clip→NDC→window (piksel) uzayına taşır,
        - imleç piksellerini üçgen içinde test eder,
        - perspektif-düzeltmeli barycentrik ile 3B noktayı geri kurar.
        Döner: (p_surface[3], normal[3], p_draw[3]) veya (None, None, None)
        """
        # 1) V,F
        if isinstance(geom, trimesh.Trimesh):
            V = np.asarray(geom.vertices, dtype=np.float64)
            F = np.asarray(geom.faces, dtype=np.int32).reshape(-1, 3)
        else:
            V = np.asarray(geom[0], dtype=np.float64)
            F = np.asarray(geom[1], dtype=np.int32).reshape(-1, 3)

        if V.size == 0 or F.size == 0 or MV is None or P is None or vp is None:
            return None, None, None

        MV = np.asarray(MV, dtype=np.float64).reshape(4, 4)
        P  = np.asarray(P,  dtype=np.float64).reshape(4, 4)
        vp = np.asarray(vp, dtype=np.int64).reshape(4)  # [x,y,w,h]
        MVP = P @ MV

        def to_clip(v3):
            v4 = np.array([v3[0], v3[1], v3[2], 1.0], dtype=np.float64)
            return MVP @ v4

        def to_window(c4):
            w = c4[3]
            if abs(w) < 1e-12:
                return None
            ndc = c4[:3] / w
            sx = vp[0] + (ndc[0] * 0.5 + 0.5) * vp[2]
            sy = vp[1] + (ndc[1] * 0.5 + 0.5) * vp[3]
            return sx, sy, ndc[2], w  # x,y,z_ndc, clip_w

        def edge(ax, ay, bx, by, px, py):
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

        best = None
        mx, my = float(mouse_win_x), float(mouse_win_y)

        for tri in F:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            v0, v1, v2 = V[i0], V[i1], V[i2]

            c0, c1, c2 = to_clip(v0), to_clip(v1), to_clip(v2)
            sw0, sw1, sw2 = to_window(c0), to_window(c1), to_window(c2)
            if (sw0 is None) or (sw1 is None) or (sw2 is None):
                continue

            x0, y0, z0, cw0 = sw0
            x1, y1, z1, cw1 = sw1
            x2, y2, z2, cw2 = sw2

            area = edge(x0, y0, x1, y1, x2, y2)
            if abs(area) < 1e-20:
                continue

            wA = edge(x1, y1, x2, y2, mx, my) / area
            wB = edge(x2, y2, x0, y0, mx, my) / area
            wC = 1.0 - wA - wB
            if (wA < -1e-6) or (wB < -1e-6) or (wC < -1e-6):
                continue  # dışarıda

            # perspektif-düzeltmeli barycentrik
            tA, tB, tC = wA / cw0, wB / cw1, wC / cw2
            s = tA + tB + tC
            if abs(s) < 1e-20:
                continue
            bA, bB, bC = tA / s, tB / s, tC / s

            hit_p = (v0 * bA) + (v1 * bB) + (v2 * bC)

            n = np.cross(v1 - v0, v2 - v0)
            nn = np.linalg.norm(n)
            if nn < 1e-20:
                continue
            n = n / nn

            z_ndc = (z0 * bA) + (z1 * bB) + (z2 * bC)
            p_draw = hit_p + n * eps_draw

            if (best is None) or (z_ndc < best[0]):  # öndeki üçgen
                best = (z_ndc, hit_p, n, p_draw)

        if best is None:
            return None, None, None
        _, hit_p, n, p_draw = best
        return hit_p, n, p_draw

    # ----- RAY TABANLI PICK (Möller–Trumbore) -----
    @staticmethod
    def pick_point(geom, ray_origin, ray_dir, eps_draw: float = 3e-3):
        """
        Ray-üçgen kesişimi ile en yakın yüzeyi bulur.
        Döner: (p_surface[3], normal[3], p_draw[3]) veya (None, None, None)
        """
        V, F = SectionService._as_VF(geom)
        if V.size == 0 or F.size == 0:
            return None, None, None

        ro = np.asarray(ray_origin, dtype=np.float64).reshape(3)
        rd = np.asarray(ray_dir,    dtype=np.float64).reshape(3)
        nrd = np.linalg.norm(rd)
        if nrd < 1e-20:
            return None, None, None
        rd = rd / nrd

        t_best = np.inf
        best_p = None
        best_n = None

        for tri in F:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            v0, v1, v2 = V[i0], V[i1], V[i2]

            # Möller–Trumbore
            e1 = v1 - v0
            e2 = v2 - v0
            pvec = np.cross(rd, e2)
            det = np.dot(e1, pvec)
            if abs(det) < 1e-12:
                continue
            inv_det = 1.0 / det
            tvec = ro - v0
            u = np.dot(tvec, pvec) * inv_det
            if u < -1e-8 or u > 1.0 + 1e-8:
                continue
            qvec = np.cross(tvec, e1)
            v = np.dot(rd, qvec) * inv_det
            if v < -1e-8 or (u + v) > 1.0 + 1e-8:
                continue
            t = np.dot(e2, qvec) * inv_det
            if t <= 1e-8:
                continue  # geride ya da çok yakın

            if t < t_best:
                t_best = t
                best_p = ro + rd * t
                n = np.cross(e1, e2)
                nn = np.linalg.norm(n)
                best_n = (n / nn) if nn > 0 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

        if best_p is None:
            return None, None, None

        p_draw = best_p + best_n * eps_draw
        return best_p, best_n, p_draw

    # ----- KESİT ALMA -----
    @staticmethod
    def compute_section(mesh: trimesh.Trimesh, point: np.ndarray, axis: str):
        """
        Verilen noktadan geçen eksene (x/y/z) dik düzlemle kesit alır.
        Döner: paths_3d: [np.ndarray(K,3), ...]
        """
        if mesh is None or point is None:
            return []
        axis = (axis or "z").lower()
        if axis == "x":
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif axis == "y":
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        section = mesh.section(plane_origin=np.asarray(point, dtype=np.float64), plane_normal=normal)
        if section is None:
            return []

        paths_3d = []
        try:
            planar, T = section.to_planar()
            T_inv = np.linalg.inv(T)
            # Dış halkaları dahil et
            for poly in getattr(planar, "polygons_full", []):
                ring2d = np.array(poly.exterior.coords, dtype=np.float64)
                ones = np.ones((ring2d.shape[0], 1))
                ring2d_h = np.hstack([ring2d, np.zeros((ring2d.shape[0], 1)), ones])
                ring3d_h = (T_inv @ ring2d_h.T).T
                ring3d = ring3d_h[:, :3] / ring3d_h[:, 3:4]
                paths_3d.append(ring3d)
        except Exception:
            # Bazı sürümlerde entities/vertices üzerinden
            try:
                for ent in section.entities:
                    if hasattr(ent, "points"):
                        pts = section.vertices[ent.points]
                        paths_3d.append(np.asarray(pts, dtype=np.float64))
            except Exception:
                pass
        return paths_3d

    # ----- TXT'ye dışa aktar -----
    @staticmethod
    def export_paths(paths, filepath: str) -> bool:
        if not paths:
            return False
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# section boundary points (x y z)\n")
            for i, poly in enumerate(paths, 1):
                f.write(f"# loop {i}\n")
                for p in poly:
                    f.write(f"{p[0]:.9f} {p[1]:.9f} {p[2]:.9f}\n")
        return True
