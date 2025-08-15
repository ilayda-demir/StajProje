# adapters/ui_adapter.py
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog,
    QHBoxLayout, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt

from adapters.opengl_viewer import OpenGLViewer
from core.section_service import SectionService

# ⬇️ ESKİ import (HATALI) → from workers.model_loader_worker import ModelLoaderWorker
# ⬇️ YENİ: ayrı process ile donmasız yükleme
from workers.process_model_loader_worker import ProcessModelLoaderWorker
# Kesiti arka planda hesaplamak için
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

        # Worker referansları
        self._proc_loader = None   # ProcessModelLoaderWorker
        self._sec_worker = None    # SectionWorker

        # --- UI öğeleri ---
        self.load_button = QPushButton("3D Model Yükle")
        self.load_button.clicked.connect(self.load_model)

        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["x", "y", "z"])
        self.axis_combo.setCurrentText("z")
        self.axis_combo.currentTextChanged.connect(self._on_axis_changed)

        self.pick_info = QPushButton("Nokta Seç (Ctrl + Sol Tık)")
        self.pick_info.setEnabled(False)  # sadece bilgi

        self.slice_button = QPushButton("Kesit Al")
        self.slice_button.clicked.connect(self.compute_section)

        self.export_button = QPushButton("TXT'ye Aktar")
        self.export_button.clicked.connect(self.export_txt)

        topbar = QHBoxLayout()
        topbar.addWidget(self.load_button)
        topbar.addWidget(self.axis_combo)
        topbar.addWidget(self.slice_button)
        topbar.addWidget(self.export_button)
        topbar.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(topbar)
        layout.addWidget(self.viewer)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Viewer içi tıklama: Ctrl+LMB ile pick
        self.viewer.mousePressEvent = self._wrap_mouse_press(self.viewer.mousePressEvent)

        # Ok tuşları ile nişangâh
        self._aim_px = None   # (mx, my) piksel
        self._aim_step = 2
        self._hook_viewer_keypress()

    # ---------------- UI Events ----------------
    def _wrap_mouse_press(self, original_handler):
        def handler(event):
            if (event.buttons() & Qt.LeftButton) and (event.modifiers() & Qt.ControlModifier):
                # 1) Piksel-eksiksiz: derinlikten oku
                p_depth = self.viewer.pick_point_from_qt(event.pos())
                if p_depth is not None:
                    # Normal şart değil; sadece görsel için nokta yeterli.
                    # İstersen projeksiyon tabanlı normal de alabilirsin (aşağıdaki fallback ile).
                    self._picked_point = p_depth
                    self._picked_normal = None
                    self.viewer.set_pick_point(p_depth, normal=None)
                    event.accept()
                    return

                # 2) Fallback: projeksiyon tabanlı barycentrik (hit bulamazsa diye)
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

    # ---------------- Model Load (Process) ----------------
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "3D Model Seç", "", "3D Modeller (*.obj *.stl)")
        if not file_path:
            return

        self.load_button.setEnabled(False)
        print("[DBG] load_model: seçilen dosya:", file_path)

        # Ayrı PROCESS ile yükle → UI donmaz
        self._proc_loader = ProcessModelLoaderWorker(file_path)
        self._proc_loader.loaded.connect(self._on_model_loaded)
        self._proc_loader.error.connect(self._on_model_load_error)
        self._proc_loader.start()

    def _on_model_loaded(self, vertices, faces):
        print("[DBG] on_model_loaded: V/F alındı → viewer’a aktarılıyor")
        self.viewer.load_model_data(vertices, faces)

        # Önceki seçim/kesitleri sıfırla
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
        # Kesiti arka planda hesapla
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

    # ---------------- Export ----------------
    def export_txt(self):
        if not getattr(self.viewer, "_section_paths", []):
            return
        path, _ = QFileDialog.getSaveFileName(self, "TXT kaydet", "", "Metin Dosyası (*.txt)")
        if not path:
            return
        SectionService.export_paths(self.viewer._section_paths, path)

    # ---------------- Aim helpers ----------------
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
            my += step  # alt-orijin -> yukarı artar
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
                my += step  # bottom-left origin
            elif k == Qt.Key_Down:
                my -= step
            elif k in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
                self._select_at_screen_px(mx, my)
                return
            elif k == Qt.Key_C:  # merkeze sıfırla
                self._ensure_aim_center()
                mx, my = self._aim_px
                self._select_at_screen_px(mx, my)
                return
            else:
                return original(ev)

            # viewport sınırlarına sıkıştır
            vx, vy, vw, vh = map(int, self.viewer._cached_vp)
            mx = max(vx, min(vx + vw - 1, int(mx)))
            my = max(vy, min(vy + vh - 1, int(my)))

            self._aim_px = (mx, my)
            self._select_at_screen_px(mx, my)

        self.viewer.keyPressEvent = on_key

    # ---------------- Temiz kapanış ----------------
    def closeEvent(self, event):
        try:
            if self._proc_loader is not None:
                try:
                    self._proc_loader.cancel()
                except Exception:
                    pass
                self._proc_loader.wait()
                self._proc_loader = None
        except Exception:
            pass

        try:
            if self._sec_worker is not None:
                self._sec_worker.quit()
                self._sec_worker.wait()
                self._sec_worker = None
        except Exception:
            pass

        super().closeEvent(event)
