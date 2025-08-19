# adapters/model_loader.py
import trimesh
import numpy as np
from pathlib import Path

class ModelLoader:
    def __init__(self):
        # Original and normalized meshes
        self.mesh = None          # original (trimesh.Trimesh)
        self.mesh_norm = None     # normalized copy (trimesh.Trimesh)

        # References for the original space
        self.center = None        # (3,)
        self.scale = None         # float

        # Cache: original / normalized vertex-face
        self._V_orig = None       # (N,3) float64
        self._F_orig = None       # (M,3) int32
        self._V_norm = None       # (N,3) float64
        self._F_norm = None       # (M,3) int32

    def load_model(self, file_path: str) -> bool:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            return False

        try:
            # Some STL/OBJ files may return a Scene; enforce 'mesh' and merge if Scene
            m = trimesh.load(file_path, force='mesh', process=False)
            if isinstance(m, trimesh.Scene):
                # Merge geometries if there are multiple
                m = trimesh.util.concatenate(tuple(m.geometry.values()))
            if not isinstance(m, trimesh.Trimesh):
                print(f"[ERROR] Expected a Trimesh Mesh, but got: {type(m)}")
                return False

            self.mesh = m

            # Original V/F
            V = np.asarray(m.vertices, dtype=np.float64)
            F = np.asarray(m.faces, dtype=np.int32).reshape(-1, 3)

            # Rarely, indices may be 1-based; ensure safety
            if F.min() == 1:
                F = F - 1

            # Check index range
            if F.min() < 0 or F.max() >= len(V):
                print(f"[WARNING] Face index out of range: min={F.min()} max={F.max()} vs len(V)={len(V)}")

            self._V_orig = V
            self._F_orig = F

            # Normalization parameters
            vmin = V.min(axis=0)
            vmax = V.max(axis=0)
            self.center = (vmin + vmax) / 2.0
            self.scale = np.linalg.norm(vmax - vmin) or 1.0

            # Normalized V
            Vn = (V - self.center) / self.scale * 2.0
            self._V_norm = Vn
            self._F_norm = F

            # Normalized mesh
            self.mesh_norm = trimesh.Trimesh(vertices=Vn, faces=F, process=False)

            print(f"[OK] Model loaded: {file_path}")
            print(f"     Vertex count: {len(V)}")
            print(f"     Face count:   {len(F)}")
            return True

        except Exception as e:
            print(f"[ERROR] Model could not be loaded: {e}")
            return False

    # Provide normalized data to the viewer
    def get_vertices(self):
        return self._V_norm

    def get_faces(self):
        return self._F_norm

    # (Optional) Access to original data if needed:
    def get_vertices_original(self):
        return self._V_orig

    def get_faces_original(self):
        return self._F_orig

    # (Optional) Transformation helpers:
    def to_original(self, p3: np.ndarray):
        """From normalized space (as seen on screen) to original space."""
        if p3 is None or self.center is None or self.scale is None:
            return None
        return self.center + np.asarray(p3, dtype=np.float64) * (self.scale / 2.0)

    def to_normalized(self, p3: np.ndarray):
        """From original space to normalized space (as seen on screen)."""
        if p3 is None or self.center is None or self.scale is None:
            return None
        return (np.asarray(p3, dtype=np.float64) - self.center) / self.scale * 2.0
