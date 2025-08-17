# adapters/model_loader_adapter.py
import numpy as np
from typing import Tuple
from core.ports import ModelLoaderPort
from adapters.model_loader import ModelLoader

class TrimeshModelLoaderAdapter(ModelLoaderPort):
    def load(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        loader = ModelLoader()
        ok = loader.load_model(file_path)
        if not ok:
            raise RuntimeError(f"Model yüklenemedi: {file_path}")
        V = loader.get_vertices().astype(np.float32, copy=False)
        F = loader.get_faces().astype(np.uint32, copy=False)
        return V, F
