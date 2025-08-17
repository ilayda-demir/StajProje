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
        # Kopyasız cast – büyük meshlerde gereksiz RAM kullanımını önler
        self.vertices = None if vertices is None else np.asarray(vertices, dtype=np.float64)
        self.faces = None if faces is None else np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        self.pick_point = None if pick_point is None else np.asarray(pick_point, dtype=np.float64).reshape(3)
        self.axis = (axis or "z").lower()

    def _validate_inputs(self):
        if self.vertices is None or self.faces is None:
            raise ValueError("Geçerli mesh verisi yok (vertices/faces None).")
        if self.vertices.size == 0 or self.faces.size == 0:
            raise ValueError("Geçerli mesh verisi yok (vertices/faces boş).")

        if self.pick_point is None or self.pick_point.shape[0] != 3:
            raise ValueError("Geçerli bir pick noktası (x,y,z) verilmelidir.")

        if not np.all(np.isfinite(self.vertices)):
            raise ValueError("Vertices içinde sonlu olmayan (NaN/Inf) değerler var.")
        if not np.all(np.isfinite(self.pick_point)):
            raise ValueError("Pick noktası içinde sonlu olmayan (NaN/Inf) değerler var.")

        if self.axis not in ("x", "y", "z"):
            raise ValueError(f"Eksen hatalı: '{self.axis}'. 'x','y' veya 'z' olmalı.")

    def run(self):
        try:
            # İptal kontrolü
            if self.isInterruptionRequested():
                return

            # 1) Giriş doğrulama
            self._validate_inputs()

            # 2) Mesh oluştur (viewer ile aynı normalize uzayda)
            # process=False -> trimesh iç işleme yapmasın (tepe/normal düzeltme yapmadan olduğu gibi al)
            mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)

            # Debug: sınırlar ve pick noktası
            try:
                bounds = mesh.bounds if hasattr(mesh, "bounds") else None
                print(f"[SEC] mesh bounds: {bounds}, pick_point: {self.pick_point}, axis: {self.axis}")
            except Exception:
                pass

            if self.isInterruptionRequested():
                return

            # 3) Kesit hesapla
            paths = SectionService.compute_section(mesh, self.pick_point, self.axis)

            # 4) Sonucu yayınla (boş olabilir → UI bunu handle ediyor)
            self.finished.emit(paths or [])

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"Kesit hesabı sırasında hata: {e}\n{tb}")
