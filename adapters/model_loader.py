import trimesh
import numpy as np
from pathlib import Path

class ModelLoader:
    def __init__(self):
        self.mesh = None
        self.vertices = None
        self.faces = None

    def load_model(self, file_path: str) -> bool:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"[HATA] Dosya bulunamadı: {file_path}")
            return False
        try:
            self.mesh = trimesh.load(file_path)
            self.vertices = np.array(self.mesh.vertices)
            self.faces = np.array(self.mesh.faces)
            print(f"[OK] Model yüklendi: {file_path}")
            print(f"     Vertex sayısı: {len(self.vertices)}")
            print(f"     Face sayısı: {len(self.faces)}")
            return True
        except Exception as e:
            print(f"[HATA] Model yüklenemedi: {e}")
            return False

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces
