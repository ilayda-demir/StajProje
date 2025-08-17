# adapters/ui_adapter.py

# en üst importlara ekleyin:
from core.use_cases import LoadModelUseCase, PickPointUseCase, ComputeSectionUseCase, ExportSectionUseCase
from core.ports import ViewerPort, SectionComputerPort, SectionExporterPort, ModelLoaderPort
from adapters.viewer_adapter import QtViewerAdapter
from adapters.section_adapter import TrimeshSectionAdapter
from adapters.txt_exporter import TxtExporter
from adapters.model_loader_adapter import TrimeshModelLoaderAdapter

from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog,
    QHBoxLayout, QComboBox, QMessageBox, QLabel, QSlider, QDoubleSpinBox
)
from PyQt5.QtCore import Qt

from adapters.opengl_viewer import OpenGLViewer
from core.section_service import SectionService

from workers.process_model_loader_worker import ProcessModelLoaderWorker
from workers.section_worker import SectionWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model Görüntüleyici")
        self.setGeometry(100, 100, 900, 700)

        self.viewer = OpenGLViewer()
        # Ports / Adapters
        self.viewer_port: ViewerPort = QtViewerAdapter(self.viewer)
        self.section_port: SectionComputerPort = TrimeshSectionAdapter()
        self.exporter_port: SectionExporterPort = TxtExporter()
        self.loader_port: ModelLoaderPort = TrimeshModelLoaderAdapter()

        # Use-cases
        self.uc_load = LoadModelUseCase(self.loader_port, self.viewer_port)
        self.uc_pick = PickPointUseCase(self.viewer_port)
        self.uc_section = ComputeSectionUseCase(self.section_port, self.viewer_port)
        self.uc_export = ExportSectionUseCase(self.exporter_port)

        self.section = SectionService()
        self._picked_point = None
        self._picked_normal = None
        self._axis = "z"

        self._proc_loader = None
        self._sec_worker = None

        # --- Üst bar ---
        self.load_button = QPushButton("3D Model Yükle")
        self.load_button.clicked.connect(self.load_model)

        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["x", "y", "z"])
        self.axis_combo.setCurrentText("z")
        self.axis_combo.currentTextChanged.connect(self._on_axis_changed)

        self.slice_button = QPushButton("Kesit Al")
        self.slice_button.clicked.connect(self.compute_section)

        self.export_button = QPushButton("TXT'ye Aktar")
        self.export_button.clicked.connect(self.export_txt)

        # --- Şeffaflık kontrolü (üst barda) ---
        self.alpha_label = QLabel("Şeffaflık:")
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(self.viewer.mesh_alpha)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setFixedWidth(110)
        self.alpha_slider.setValue(int(round(self.viewer.mesh_alpha * 100)))

        self.alpha_spin.valueChanged.connect(self._on_alpha_spin_changed)
        self.alpha_slider.valueChanged.connect(self._on_alpha_slider_changed)

        topbar = QHBoxLayout()
        topbar.addWidget(self.load_button)
        topbar.addWidget(self.axis_combo)
        topbar.addWidget(self.slice_button)
        topbar.addWidget(self.export_button)
        topbar.addSpacing(12)
        topbar.addWidget(self.alpha_label)
        topbar.addWidget(self.alpha_spin)
        topbar.addWidget(self.alpha_slider)
        topbar.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(topbar)
        layout.addWidget(self.viewer)



        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Ctrl + Sol Tık ile pick
        self.viewer.mousePressEvent = self._wrap_mouse_press(self.viewer.mousePressEvent)



    # ---------------- UI Events ----------------
    def _wrap_mouse_press(self, original_handler):
        def handler(event):
            if (event.buttons() & Qt.LeftButton) and (event.modifiers() & Qt.ControlModifier):
                p_depth = self.viewer.pick_point_from_qt(event.pos())
                if p_depth is not None:
                    self._picked_point = p_depth
                    self._picked_normal = None
                    self.uc_pick(p_depth, None)

                    event.accept()
                    return

                vp = self.viewer._cached_vp
                MV = self.viewer._cached_model
                P = self.viewer._cached_proj
                if vp is None or MV is None or P is None:
                    event.accept()
                    return

                dpr = float(self.viewer.devicePixelRatioF())
                mx = event.pos().x() * dpr
                my = (float(vp[3]) - 1.0) - (event.pos().y() * dpr)

                p_surface, normal, p_draw = self.section.pick_point_projective(
                    (self.viewer.vertices, self.viewer.faces),
                    MV, P, vp, mx, my, eps_draw=3e-3
                )
                if p_surface is not None:
                    self._picked_point = p_surface
                    self._picked_normal = normal
                    self.uc_pick(p_draw, normal)

                event.accept()
                return

            return original_handler(event)
        return handler

    def _on_axis_changed(self, s):
        self._axis = s.lower()

    def _on_alpha_spin_changed(self, val: float):
        self.viewer_port.set_alpha(float(val))
        self.alpha_slider.blockSignals(True)
        self.alpha_slider.setValue(int(round(float(val) * 100)))
        self.alpha_slider.blockSignals(False)

    def _on_alpha_slider_changed(self, v: int):
        a = v / 100.0
        self.viewer_port.set_alpha(a)
        self.alpha_spin.blockSignals(True)
        self.alpha_spin.setValue(a)
        self.alpha_spin.blockSignals(False)

    # ---------------- Model Load (Process) ----------------
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "3D Model Seç", "", "3D Modeller (*.obj *.stl)")
        if not file_path:
            return

        self.load_button.setEnabled(False)
        print("[DBG] load_model: seçilen dosya:", file_path)

        self._proc_loader = ProcessModelLoaderWorker(file_path)
        self._proc_loader.loaded.connect(self._on_model_loaded)
        self._proc_loader.error.connect(self._on_model_load_error)
        self._proc_loader.start()

    def _on_model_loaded(self, vertices, faces):
        print("[DBG] on_model_loaded: V/F alındı → viewer’a aktarılıyor")
        self.viewer_port.show_mesh(vertices, faces)
        self._picked_point = None
        self._picked_normal = None
        self.viewer_port.clear_pick_and_sections()

        self.load_button.setEnabled(True)
        self._proc_loader = None

    def _on_model_load_error(self, msg):
        print("[ERR] Model yükleme hatası:", msg)
        QMessageBox.critical(self, "Model Yükleme Hatası", msg or "Bilinmeyen hata")
        self.load_button.setEnabled(True)
        self._proc_loader = None

    # ---------------- Section (Thread) ----------------
    def compute_section(self):
        if self.viewer.vertices is None or self._picked_point is None:
            return

        self.slice_button.setEnabled(False)
        self._sec_worker = SectionWorker(
            self.viewer.vertices, self.viewer.faces, self._picked_point, self._axis
        )
        self._sec_worker.finished.connect(self._on_section_done)
        self._sec_worker.error.connect(self._on_section_err)
        self._sec_worker.start()

    def _on_section_done(self, paths):
        self.viewer_port.show_sections(paths)
        self.slice_button.setEnabled(True)
        self._sec_worker = None

    def _on_section_err(self, msg):
        QMessageBox.critical(self, "Kesit Hatası", msg or "Bilinmeyen hata")
        self.slice_button.setEnabled(True)
        self._sec_worker = None

    def export_txt(self):
        paths = getattr(self.viewer, "_section_paths", [])
        if not paths:
            QMessageBox.information(self, "Bilgi", "Kesit verisi yok.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "TXT olarak kaydet", "", "Metin Dosyası (*.txt)")
        if not file_path:
            return
        try:
            self.uc_export(paths, file_path)
            QMessageBox.information(self, "Başarılı", "TXT olarak kaydedildi.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Kaydedilemedi: {e}")








