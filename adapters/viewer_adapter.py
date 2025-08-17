# adapters/viewer_adapter.py
import numpy as np
from typing import List, Optional
from adapters.opengl_viewer import OpenGLViewer
from core.ports import ViewerPort

class QtViewerAdapter(ViewerPort):
    def __init__(self, ogl: OpenGLViewer):
        self.ogl = ogl

    def show_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.ogl.load_model_data(vertices, faces)

    def clear_pick_and_sections(self) -> None:
        self.ogl.set_pick_point(None)
        self.ogl.set_section_paths([])

    def show_pick(self, p3: Optional[np.ndarray], normal: Optional[np.ndarray]) -> None:
        self.ogl.set_pick_point(p3, normal=normal)

    def show_sections(self, paths: List[np.ndarray]) -> None:
        self.ogl.set_section_paths(paths)

    def set_alpha(self, a: float) -> None:
        self.ogl.set_mesh_alpha(a)
