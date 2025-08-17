# adapters/section_adapter.py
import numpy as np
from typing import List
from core.ports import SectionComputerPort
from core.section_service import SectionService
import trimesh

class TrimeshSectionAdapter(SectionComputerPort):
    def compute(self, vertices: np.ndarray, faces: np.ndarray, pick_point: np.ndarray, axis: str) -> List[np.ndarray]:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return SectionService.compute_section(mesh, pick_point, axis)
