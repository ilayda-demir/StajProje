# adapters/model_loader.py
import trimesh
import numpy as np
from pathlib import Path

class ModelLoader:
    def __init__(self):
        # Orijinal ve normalize meshler
        self.mesh = None          # orijinal (trimesh.Trimesh)
        self.mesh_norm = None     # normalize kopya (trimesh.Trimesh)

        # Orijinal uzay için referanslar
        self.center = None        # (3,)
        self.scale = None         # float

        # Cache: orijinal / normalize vertex-face
        self._V_orig = None       # (N,3) float64
        self._F_orig = None       # (M,3) int32
        self._V_norm = None       # (N,3) float64
        self._F_norm = None       # (M,3) int32

    def load_model(self, file_path: str) -> bool:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"[HATA] Dosya bulunamadı: {file_path}")
            return False

        try:
            # Bazı STL/OBJ dosyaları Scene döndürebilir; force='mesh' ve Scene birleştirme
            m = trimesh.load(file_path, force='mesh', process=False)
            if isinstance(m, trimesh.Scene):
                # Birden fazla geometri varsa birleştir
                m = trimesh.util.concatenate(tuple(m.geometry.values()))
            if not isinstance(m, trimesh.Trimesh):
                print(f"[HATA] Trimesh Mesh bekleniyordu, ama: {type(m)}")
                return False

            self.mesh = m

            # Orijinal V/F
            V = np.asarray(m.vertices, dtype=np.float64)
            F = np.asarray(m.faces, dtype=np.int32).reshape(-1, 3)

            # Çok nadir 1-bazlı gelebilir; garanti altına al
            if F.min() == 1:
                F = F - 1

            # İndeks aralığı kontrolü
            if F.min() < 0 or F.max() >= len(V):
                print(f"[UYARI] Face index out of range: min={F.min()} max={F.max()} vs len(V)={len(V)}")

            self._V_orig = V
            self._F_orig = F

            # Normalize parametreleri
            vmin = V.min(axis=0)
            vmax = V.max(axis=0)
            self.center = (vmin + vmax) / 2.0
            self.scale = np.linalg.norm(vmax - vmin) or 1.0

            # Normalize edilmiş V
            Vn = (V - self.center) / self.scale * 2.0
            self._V_norm = Vn
            self._F_norm = F

            # Normalize mesh
            self.mesh_norm = trimesh.Trimesh(vertices=Vn, faces=F, process=False)

            print(f"[OK] Model yüklendi: {file_path}")
            print(f"     Vertex sayısı: {len(V)}")
            print(f"     Face sayısı:   {len(F)}")
            return True

        except Exception as e:
            print(f"[HATA] Model yüklenemedi: {e}")
            return False

    # Viewer'a normalize veriyi veriyoruz
    def get_vertices(self):
        return self._V_norm

    def get_faces(self):
        return self._F_norm

    # (İsteğe bağlı) Orijinal veriye erişim gerekirse:
    def get_vertices_original(self):
        return self._V_orig

    def get_faces_original(self):
        return self._F_orig

    # (İsteğe bağlı) Dönüşüm yardımcıları:
    def to_original(self, p3: np.ndarray):
        """Normalize (ekranda görünen) uzaydan orijinal uzaya."""
        if p3 is None or self.center is None or self.scale is None:
            return None
        return self.center + np.asarray(p3, dtype=np.float64) * (self.scale / 2.0)

    def to_normalized(self, p3: np.ndarray):
        """Orijinal uzaydan normalize (ekranda görünen) uzaya."""
        if p3 is None or self.center is None or self.scale is None:
            return None
        return (np.asarray(p3, dtype=np.float64) - self.center) / self.scale * 2.0
