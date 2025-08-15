# adapters/ui_adapter.py
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

        # Ok tuşları ile nişangâh (altyapı korunuyor)
        self._aim_px = None
        self._aim_step = 2
        self._hook_viewer_keypress()

    # ---------------- UI Events ----------------
    def _wrap_mouse_press(self, original_handler):
        def handler(event):
            if (event.buttons() & Qt.LeftButton) and (event.modifiers() & Qt.ControlModifier):
                p_depth = self.viewer.pick_point_from_qt(event.pos())
                if p_depth is not None:
                    self._picked_point = p_depth
                    self._picked_normal = None
                    self.viewer.set_pick_point(p_depth, normal=None)
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
                    self.viewer.set_pick_point(p_draw, normal=normal)
                event.accept()
                return

            return original_handler(event)
        return handler

    def _on_axis_changed(self, s):
        self._axis = s.lower()

    # ---------------- Şeffaflık sync ----------------
    def _on_alpha_spin_changed(self, val: float):
        self.viewer.set_mesh_alpha(float(val))
        self.alpha_slider.blockSignals(True)
        self.alpha_slider.setValue(int(round(float(val) * 100)))
        self.alpha_slider.blockSignals(False)

    def _on_alpha_slider_changed(self, v: int):
        a = v / 100.0
        self.viewer.set_mesh_alpha(a)
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
        self.viewer.load_model_data(vertices, faces)

        self._picked_point = None
        self._picked_normal = None
        self.viewer.set_pick_point(None)
        self.viewer.set_section_paths([])

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
        self.viewer.set_section_paths(paths)
        self.slice_button.setEnabled(True)
        self._sec_worker = None

    def _on_section_err(self, msg):
        QMessageBox.critical(self, "Kesit Hatası", msg or "Bilinmeyen hata")
        self.slice_button.setEnabled(True)
        self._sec_worker = None

    # ---------------- TXT export ----------------
    def export_txt(self):
        if not getattr(self.viewer, "_section_paths", []):
            QMessageBox.information(self, "Bilgi", "Kesit verisi yok.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "TXT olarak kaydet", "", "Metin Dosyası (*.txt)")
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for poly in self.viewer._section_paths:
                    for p in poly:
                        f.write(f"{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}\n")
                    f.write("\n")
            QMessageBox.information(self, "Başarılı", "TXT olarak kaydedildi.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Kaydedilemedi: {e}")

    # ---------------- Aim helpers (mevcut davranış) ----------------
    def _ensure_aim_center(self):
        vp = self.viewer._cached_vp
        if vp is None:
            return None
        mx = vp[0] + vp[2] * 0.5
        my = vp[1] + vp[3] * 0.5
        self._aim_px = (mx, my)
        return self._aim_px

    def _select_at_screen_px(self, mx, my):
        ray_o, ray_d = self.viewer.compute_ray_from_window_pixels(mx, my)
        if ray_o is None or ray_d is None:
            return
        p_surface, normal, p_draw = self.section.pick_point(
            (self.viewer.vertices, self.viewer.faces), ray_o, ray_d
        )
        if p_surface is not None:
            self._picked_point, self._picked_normal = p_surface, normal
            self.viewer.set_pick_point(p_draw, normal=normal)

    def keyPressEvent(self, ev):
        if self._aim_px is None and self.viewer._cached_vp is not None:
            self._ensure_aim_center()
        if self._aim_px is None:
            return super().keyPressEvent(ev)

        mx, my = self._aim_px
        step = self._aim_step
        if ev.key() == Qt.Key_Left:
            mx -= step
        elif ev.key() == Qt.Key_Right:
            mx += step
        elif ev.key() == Qt.Key_Up:
            my += step
        elif ev.key() == Qt.Key_Down:
            my -= step
        elif ev.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self._select_at_screen_px(mx, my)
            return
        else:
            return super().keyPressEvent(ev)

        self._aim_px = (mx, my)
        self._select_at_screen_px(mx, my)

    def _hook_viewer_keypress(self):
        original = self.viewer.keyPressEvent
        def on_key(ev):
            if self.viewer._cached_vp is None:
                return original(ev)

            if self._aim_px is None:
                self._ensure_aim_center()

            mx, my = self._aim_px
            step = self._aim_step
            if ev.modifiers() & Qt.ShiftModifier:
                step *= 10

            k = ev.key()
            if k == Qt.Key_Left:
                mx -= step
            elif k == Qt.Key_Right:
                mx += step
            elif k == Qt.Key_Up:
                my += step
            elif k == Qt.Key_Down:
                my -= step
            elif k in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
                self._select_at_screen_px(mx, my)
                return
            elif k == Qt.Key_C:
                self._ensure_aim_center()
                mx, my = self._aim_px
                self._select_at_screen_px(mx, my)
                return
            else:
                return original(ev)

            vx, vy, vw, vh = map(int, self.viewer._cached_vp)
            mx = max(vx, min(vx + vw - 1, int(mx)))
            my = max(vy, min(vy + vh - 1, int(my)))

            self._aim_px = (mx, my)
            self._select_at_screen_px(mx, my)

        self.viewer.keyPressEvent = on_key
