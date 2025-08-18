# adapters/ui_adapter.py

import numpy as np

from PyQt5.QtCore import Qt, QLocale
from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog,
    QHBoxLayout, QComboBox, QMessageBox, QLabel, QSlider, QDoubleSpinBox
)

from core.use_cases import LoadModelUseCase, PickPointUseCase, ComputeSectionUseCase, ExportSectionUseCase
from core.ports import ViewerPort, SectionComputerPort, SectionExporterPort, ModelLoaderPort
from adapters.viewer_adapter import QtViewerAdapter
from adapters.section_adapter import TrimeshSectionAdapter
from adapters.txt_exporter import TxtExporter
from adapters.model_loader_adapter import TrimeshModelLoaderAdapter

from adapters.opengl_viewer import OpenGLViewer
from core.section_service import SectionService

from workers.process_model_loader_worker import ProcessModelLoaderWorker
from workers.section_worker import SectionWorker


# ---- Yazım sırasında ara metinleri ( "-", "-.", "." vb.) kabul eden spinbox ----
class LooseDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setKeyboardTracking(False)      # yazarken anında "fixup" yapmasın
        self.setLocale(QLocale.c())          # ondalık ayırıcı = '.'

    def validate(self, text, pos):
        # Ara durumları "Intermediate" olarak kabul et
        if text in ("", "-", "+", ".", "-.", "+."):
            return (QValidator.Intermediate, text, pos)
        if text.lower() in ("e", "-e", "+e"):   # bilimsel gösterim ara durumu (opsiyonel)
            return (QValidator.Intermediate, text, pos)
        return super().validate(text, pos)


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

        # State
        self._picked_point = None
        self._picked_normal = None
        self._axis = "z"

        # Orijinal uzay bilgisi (varsa)
        self._orig_center = None      # np.ndarray(3,)
        self._orig_scale  = None      # float
        self._coord_mode  = "Normalize"  # "Normalize" | "Orijinal" (orijinal bilgi yoksa yalnızca Normalize)

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

        # --- Şeffaflık kontrolü ---
        self.alpha_label = QLabel("Şeffaflık:")
        self.alpha_spin = LooseDoubleSpinBox()
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

        # --- Koordinat uzayı seçimi ---
        self.space_label = QLabel("Uzay:")
        self.space_combo = QComboBox()
        self._refresh_space_combo()
        self.space_combo.currentTextChanged.connect(self._on_space_changed)

        # --- Manuel pick girişi ---
        self.manual_label = QLabel("Nokta (x,y,z):")
        self.x_spin = LooseDoubleSpinBox()
        self.y_spin = LooseDoubleSpinBox()
        self.z_spin = LooseDoubleSpinBox()
        for sp in (self.x_spin, self.y_spin, self.z_spin):
            sp.setRange(-1_000_000.0, 1_000_000.0)  # geniş aralık; modeli yükleyince daraltıyoruz
            sp.setDecimals(6)
            sp.setSingleStep(0.01)
            sp.setValue(0.0)

        self.apply_point_btn = QPushButton("Noktayı Uygula")
        self.apply_point_btn.clicked.connect(self._apply_manual_point)

        # --- Layout ---
        topbar = QHBoxLayout()
        topbar.addWidget(self.load_button)
        topbar.addWidget(self.axis_combo)
        topbar.addWidget(self.slice_button)
        topbar.addWidget(self.export_button)
        topbar.addSpacing(12)
        topbar.addWidget(self.alpha_label)
        topbar.addWidget(self.alpha_spin)
        topbar.addWidget(self.alpha_slider)
        topbar.addSpacing(12)
        topbar.addWidget(self.space_label)
        topbar.addWidget(self.space_combo)
        topbar.addSpacing(12)
        topbar.addWidget(self.manual_label)
        topbar.addWidget(self.x_spin)
        topbar.addWidget(self.y_spin)
        topbar.addWidget(self.z_spin)
        topbar.addWidget(self.apply_point_btn)
        topbar.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(topbar)
        layout.addWidget(self.viewer)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Ctrl + Sol Tık ile pick
        self.viewer.mousePressEvent = self._wrap_mouse_press(self.viewer.mousePressEvent)

    # ---------------- Helpers: koordinat dönüşümleri ----------------
    def _to_original(self, P):
        """Normalize → Orijinal (tek nokta veya Nx3)."""
        if self._orig_center is None or self._orig_scale is None or P is None:
            return P
        P = np.asarray(P, dtype=np.float64)
        return self._orig_center + P * (self._orig_scale / 2.0)

    def _to_normalized(self, P):
        """Orijinal → Normalize (tek nokta veya Nx3)."""
        if self._orig_center is None or self._orig_scale is None or P is None:
            return P
        P = np.asarray(P, dtype=np.float64)
        return (P - self._orig_center) / self._orig_scale * 2.0

    def _refresh_space_combo(self):
        """Orijinal bilgiye göre uzay seçeneklerini güncelle."""
        have_orig = (self._orig_center is not None) and (self._orig_scale is not None)
        current = self._coord_mode
        self.space_combo.blockSignals(True)
        self.space_combo.clear()
        self.space_combo.addItem("Normalize")
        if have_orig:
            self.space_combo.addItem("Orijinal")
        else:
            # Orijinal yoksa mod 'Normalize' kalsın
            current = "Normalize"
            self._coord_mode = "Normalize"
        self.space_combo.setCurrentText(current)
        self.space_combo.blockSignals(False)

    # ---------------- UI Events ----------------
    def _wrap_mouse_press(self, original_handler):
        def handler(event):
            if (event.buttons() & Qt.LeftButton) and (event.modifiers() & Qt.ControlModifier):
                p_depth = self.viewer.pick_point_from_qt(event.pos())
                if p_depth is not None:
                    self._picked_point = p_depth
                    self._picked_normal = None
                    self.uc_pick(p_depth, None)
                    # SpinBox'ları etkin uzaya göre doldur
                    p_show = self._to_original(p_depth) if self._coord_mode == "Orijinal" else p_depth
                    self.x_spin.setValue(float(p_show[0]))
                    self.y_spin.setValue(float(p_show[1]))
                    self.z_spin.setValue(float(p_show[2]))
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
                    p_show = self._to_original(p_draw) if self._coord_mode == "Orijinal" else p_draw
                    self.x_spin.setValue(float(p_show[0]))
                    self.y_spin.setValue(float(p_show[1]))
                    self.z_spin.setValue(float(p_show[2]))

                event.accept()
                return

            return original_handler(event)
        return handler

    def _on_axis_changed(self, s):
        self._axis = s.lower()

    def _on_space_changed(self, txt):
        self._coord_mode = txt
        # Mevcut vertex'lere göre spin aralıklarını güncelle
        if self.viewer.vertices is not None:
            self._set_spin_ranges_from_mesh(self.viewer.vertices, space=self._coord_mode)

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

    def _apply_manual_point(self):
        # Kullanıcının ekranda gördüğü uzaydan normalize'a çevirip viewer'a gönder
        px = float(self.x_spin.value())
        py = float(self.y_spin.value())
        pz = float(self.z_spin.value())
        p_in = np.array([px, py, pz], dtype=np.float64)
        p_norm = self._to_normalized(p_in) if self._coord_mode == "Orijinal" else p_in

        self._picked_point = p_norm
        self._picked_normal = None
        self.uc_pick(p_norm, None)

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

    def _on_model_loaded(self, *args):
        """
        Worker iki farklı imza ile gelebilir:
          - (V, F)
          - (V, F, C, S)  -> center & scale varsa
        """
        if len(args) == 2:
            vertices, faces = args
            center, scale = None, None
        elif len(args) >= 4:
            vertices, faces, center, scale = args[:4]
        else:
            QMessageBox.critical(self, "Model Yükleme Hatası", "Beklenmeyen yükleme çıktısı.")
            self.load_button.setEnabled(True)
            self._proc_loader = None
            return

        print("[DBG] on_model_loaded: V/F alındı → viewer’a aktarılıyor")
        self.viewer_port.show_mesh(vertices, faces)

        # Orijinal uzay bilgisi (varsa) sakla
        self._orig_center = None if center is None else np.asarray(center, dtype=np.float64)
        self._orig_scale  = None if scale  is None else float(scale)
        self._refresh_space_combo()

        # Aktif uzaya göre aralıkları belirle
        self._set_spin_ranges_from_mesh(vertices, space=self._coord_mode)

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

    # adapters/ui_adapter.py içindeki export_txt()’yi şöyle güncelle
    def export_txt(self):
        paths = getattr(self.viewer, "_section_paths", [])
        if not paths:
            QMessageBox.information(self, "Bilgi", "Kesit verisi yok.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "TXT olarak kaydet", "", "Metin Dosyası (*.txt)")
        if not file_path:
            return

        # Hangi uzayda kaydedelim? (Mevcut seçim: self._coord_mode)
        paths_to_save = paths
        try:
            if (self._coord_mode == "Orijinal"
                    and self._orig_center is not None and self._orig_scale is not None):
                # Normalize → Orijinal (Nx3 vektörel çalışır)
                paths_to_save = [self._to_original(np.asarray(poly, dtype=np.float64)) for poly in paths]
        except Exception as e:
            QMessageBox.warning(self, "Uyarı", f"Orijinale çevirilemedi, normalize kaydedilecek.\n{e}")
            paths_to_save = paths

        try:
            self.uc_export(paths_to_save, file_path)
            QMessageBox.information(self, "Başarılı", "TXT olarak kaydedildi.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Kaydedilemedi: {e}")

    # ---------------- Ranges ----------------
    def _set_spin_ranges_from_mesh(self, V: np.ndarray, space: str = "Normalize"):
        """Spin aralıklarını aktif uzaya göre ayarlar."""
        if V is None or V.size == 0:
            return
        Vn = np.asarray(V, dtype=float)
        Vuse = self._to_original(Vn) if space == "Orijinal" else Vn   # vektörel çalışır
        vmin = Vuse.min(axis=0)
        vmax = Vuse.max(axis=0)
        extent = np.maximum(vmax - vmin, 1e-6)
        pad = 0.50 * float(np.max(extent))  # geniş tampon: yazarken kısıtlamasın

        self.x_spin.setRange(float(vmin[0] - pad), float(vmax[0] + pad))
        self.y_spin.setRange(float(vmin[1] - pad), float(vmax[1] + pad))
        self.z_spin.setRange(float(vmin[2] - pad), float(vmax[2] + pad))

        step = max(1e-6, 0.005 * float(np.max(extent)))
        for sp in (self.x_spin, self.y_spin, self.z_spin):
            sp.setSingleStep(step)
            sp.setDecimals(6)
