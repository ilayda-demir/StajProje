# workers/section_worker.py
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import trimesh
import traceback
from core.section_service import SectionService


class SectionWorker(QThread):
    """
    QThread-based worker to compute cross-sections (sections) in the background.
    Usage:
      worker = SectionWorker(vertices, faces, pick_point, axis)  # axis: "x", "y" or "z"
      worker.finished.connect(on_paths)  # (paths: List[np.ndarray(N,3)])
      worker.error.connect(on_error)     # (msg: str)
      worker.start()
    """
    finished = pyqtSignal(object)  # paths: List[np.ndarray(N,3)]
    error = pyqtSignal(str)

    def __init__(self, vertices, faces, pick_point, axis: str = "z"):
        super().__init__()
        # Cast without copying – prevents unnecessary RAM usage on large meshes
        self.vertices = None if vertices is None else np.asarray(vertices, dtype=np.float64)
        self.faces = None if faces is None else np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        self.pick_point = None if pick_point is None else np.asarray(pick_point, dtype=np.float64).reshape(3)
        self.axis = (axis or "z").lower()

    def _validate_inputs(self):
        if self.vertices is None or self.faces is None:
            raise ValueError("No valid mesh data (vertices/faces are None).")
        if self.vertices.size == 0 or self.faces.size == 0:
            raise ValueError("No valid mesh data (vertices/faces are empty).")

        if self.pick_point is None or self.pick_point.shape[0] != 3:
            raise ValueError("A valid pick point (x,y,z) must be provided.")

        if not np.all(np.isfinite(self.vertices)):
            raise ValueError("Vertices contain non-finite values (NaN/Inf).")
        if not np.all(np.isfinite(self.pick_point)):
            raise ValueError("Pick point contains non-finite values (NaN/Inf).")

        if self.axis not in ("x", "y", "z"):
            raise ValueError(f"Invalid axis: '{self.axis}'. Must be 'x', 'y' or 'z'.")

    def run(self):
        try:
            # Check for cancellation
            if self.isInterruptionRequested():
                return

            # 1) Validate inputs
            self._validate_inputs()

            # 2) Build mesh (in the same normalized space as viewer)
            # process=False -> prevents trimesh from auto-fixing (keeps raw vertices/normals)
            mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)

            # Debug: bounds and pick point
            try:
                bounds = mesh.bounds if hasattr(mesh, "bounds") else None
                print(f"[SEC] mesh bounds: {bounds}, pick_point: {self.pick_point}, axis: {self.axis}")
            except Exception:
                pass

            if self.isInterruptionRequested():
                return

            # 3) Compute section
            paths = SectionService.compute_section(mesh, self.pick_point, self.axis)

            # 4) Emit result (can be empty → UI handles this)
            self.finished.emit(paths or [])

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"Error while computing section: {e}\n{tb}")
