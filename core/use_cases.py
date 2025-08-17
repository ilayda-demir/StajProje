# core/use_cases.py
from typing import List, Optional, Tuple
import numpy as np
from .ports import ModelLoaderPort, SectionComputerPort, ViewerPort, SectionExporterPort

class LoadModelUseCase:
    def __init__(self, loader: ModelLoaderPort, viewer: ViewerPort):
        self.loader = loader
        self.viewer = viewer

    def __call__(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        V, F = self.loader.load(file_path)
        self.viewer.show_mesh(V, F)
        self.viewer.clear_pick_and_sections()
        return V, F

class PickPointUseCase:
    def __init__(self, viewer: ViewerPort):
        self.viewer = viewer

    def __call__(self, p3: Optional[np.ndarray], normal: Optional[np.ndarray]) -> None:
        self.viewer.show_pick(p3, normal)

class ComputeSectionUseCase:
    def __init__(self, sectioner: SectionComputerPort, viewer: ViewerPort):
        self.sectioner = sectioner
        self.viewer = viewer

    def __call__(self, V: np.ndarray, F: np.ndarray, p3: np.ndarray, axis: str):
        paths = self.sectioner.compute(V, F, p3, axis)
        self.viewer.show_sections(paths)
        return paths

class ExportSectionUseCase:
    def __init__(self, exporter: SectionExporterPort):
        self.exporter = exporter

    def __call__(self, paths: List[np.ndarray], file_path: str) -> None:
        self.exporter.export_txt(paths, file_path)
