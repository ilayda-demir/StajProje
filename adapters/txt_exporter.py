# adapters/txt_exporter.py
from typing import List
import numpy as np
import csv
from core.ports import SectionExporterPort

class TxtExporter(SectionExporterPort):
    def export_txt(self, paths: List[np.ndarray], file_path: str) -> None:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for poly in paths:
                for p in poly:
                    writer.writerow([f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
