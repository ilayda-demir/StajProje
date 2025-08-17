# adapters/txt_exporter.py
from typing import List
import numpy as np
from core.ports import SectionExporterPort

class TxtExporter(SectionExporterPort):
    def export_txt(self, paths: List[np.ndarray], file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            for poly in paths:
                for p in poly:
                    f.write(f"{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}\n")
                f.write("\n")
