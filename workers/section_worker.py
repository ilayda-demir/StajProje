# workers/section_worker.py
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import trimesh
import traceback
from core.section_service import SectionService


class SectionWorker(QThread):
    """
    Kesit hesaplamasını (section) arka planda yapmak için QThread tabanlı worker.
    Kullanım:
      worker = SectionWorker(vertices, faces, pick_point, axis)  # axis: "x", "y" veya "z"
      worker.finished.connect(on_paths)  # (paths: List[np.ndarray(N,3)])
      worker.error.connect(on_error)     # (msg: str)
      worker.start()
    """
    finished = pyqtSignal(object)  # paths: List[np.ndarray(N,3)]
    error = pyqtSignal(str)

    def __init__(self, vertices, faces, pick_point, axis: str = "z"):
        super().__init__()
        self.vertices = np.asarray(vertices, dtype=np.float64) if vertices is not None else None
        self.faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3) if faces is not None else None
        self.pick_point = None if pick_point is None else np.asarray(pick_point, dtype=np.float64)
        self.axis = (axis or "z").lower()

    def run(self):
        try:
            if self.vertices is None or self.faces is None or self.vertices.size == 0 or self.faces.size == 0:
                raise ValueError("Geçerli mesh verisi yok (vertices/faces boş).")
            if self.pick_point is None or self.pick_point.shape[0] != 3:
                raise ValueError("Geçerli bir pick noktası (x,y,z) verilmelidir.")
            if self.axis not in ("x", "y", "z"):
                raise ValueError(f"Eksen hatalı: '{self.axis}'. 'x','y' veya 'z' olmalı.")

            # Viewer'da çizilen normalize mesh ile aynı uzayda çalışmak için process=False
            mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)

            # SectionService'i kullanarak kesit hesapla
            paths = SectionService.compute_section(mesh, self.pick_point, self.axis)

            # Sonucu ana threade gönder
            self.finished.emit(paths or [])

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"Kesit hesabı sırasında hata: {e}\n{tb}")
