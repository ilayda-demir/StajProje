# adapters/opengl_viewer.py
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QSurfaceFormat

from OpenGL.GL import *
from OpenGL.GLU import gluPerspective  # GLU perspektif için
import numpy as np


class OpenGLViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Güvenli ve tutarlı GL formatı (fixed pipeline uyumlu)
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        fmt.setVersion(2, 1)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(0)  # MSAA kapalı (Intel sürücülerinde daha stabil)
        self.setFormat(fmt)

        # Durum
        self.vertices = None
        self.faces = None
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.zoom_distance = -5.0
        self.last_mouse_pos = None
        self.setFocusPolicy(Qt.StrongFocus)

        # Seçim/çizim
        self._picked = None
        self._picked_n = None
        self._picked_ray = None
        self._section_paths = []

        # paintGL'de doldurulacak cache (ray/pick ve projeksiyon tabanlı seçim için)
        self._cached_model = None  # 4x4
        self._cached_proj  = None  # 4x4
        self._cached_vp    = None  # [x,y,w,h] (piksel)

    # ---- Model yükle (viewer için: tekrar normalize ETME) ----
    def load_model_data(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces    = np.asarray(faces, dtype=np.int32).reshape(-1, 3)

        # Kamera reset
        self.rotation_x = self.rotation_y = self.rotation_z = 0.0
        self.zoom_distance = -5.0
        self.last_mouse_pos = None

        # Seçim/kesit temizle
        self._picked = None
        self._section_paths = []
        self.update()

    # ---- GL ömür döngüsü ----
    def initializeGL(self):
        glDisable(GL_MULTISAMPLE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glClearDepth(1.0)
        glClearColor(0.12, 0.12, 0.12, 1.0)
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)

        try:
            ver = glGetString(GL_VERSION)
            ren = glGetString(GL_RENDERER)
            print("[GL] Version:", ver.decode() if ver else ver, "| Renderer:", ren.decode() if ren else ren)
        except Exception:
            pass

        # İlk karede projeksiyonu kur
        w = max(1, self.width())
        h = max(1, self.height())
        self.resizeGL(w, h)

    def resizeGL(self, w, h):
        # Viewport'u PİKSEL cinsinden aç (HiDPI doğru hizalama)
        dpr  = float(self.devicePixelRatioF())
        vp_w = max(1, int(w * dpr))
        vp_h = max(1, int(h * dpr))
        glViewport(0, 0, vp_w, vp_h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = (w / h) if h else 1.0  # aspect DIP üzerinden hesaplanır
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Kamera (zoom + dönüşler)
        glTranslatef(0.0, 0.0, self.zoom_distance)
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        glRotatef(self.rotation_z, 0.0, 0.0, 1.0)

        # --- MATRIS CACHE (event'te GL çağrısı yapmamak için) ---
        from OpenGL.GL import (
            glGetDoublev, glGetIntegerv,
            GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT
        )
        model = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj  = glGetDoublev(GL_PROJECTION_MATRIX)
        vp    = glGetIntegerv(GL_VIEWPORT)
        # OpenGL sütun-major döner → NumPy'de doğru çarpım için F-order reshape
        self._cached_model = np.array(model, dtype=np.float64).reshape((4, 4), order='F')
        self._cached_proj  = np.array(proj,  dtype=np.float64).reshape((4, 4), order='F')
        self._cached_vp    = np.array(vp,    dtype=np.int32)

        # --- Eksenler ---
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X (kırmızı)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-2.0, 0.0, 0.0)
        glVertex3f( 2.0, 0.0, 0.0)
        # Y (yeşil)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, -2.0, 0.0)
        glVertex3f(0.0,  2.0, 0.0)
        # Z (mavi)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, -2.0)
        glVertex3f(0.0, 0.0,  2.0)
        glEnd()
        glLineWidth(1.0)

        # paintGL içinde, "Mesh" bölümünün yerine:
        if self.vertices is not None and self.faces is not None:
            glColor3f(0.0, 0.6, 1.0)

            # Veriyi tek seferde GPU'ya akıtmak için client-side array kullan
            # (VBO'suz, ama glBegin/glEnd'den çok daha hızlı)
            V = np.asarray(self.vertices, dtype=np.float32, order='C')
            F = np.asarray(self.faces, dtype=np.uint32, order='C').ravel()

            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, V)
            glDrawElements(GL_TRIANGLES, F.size, GL_UNSIGNED_INT, F)
            glDisableClientState(GL_VERTEX_ARRAY)

        # --- Kesit yolları (varsa) ---
        if self._section_paths:
            glColor3f(1.0, 1.0, 0.0)
            for poly in self._section_paths:
                if poly is None or len(poly) == 0:
                    continue
                glBegin(GL_LINE_STRIP)
                for p in poly:
                    px, py, pz = map(float, p)
                    glVertex3f(px, py, pz)
                glEnd()

        # --- Seçili nokta (her zaman görünür) ---
        if self._picked is not None:
            px, py, pz = map(float, self._picked)

            # Mesh ile Z-savaşını önlemek için geçici depth off
            glDisable(GL_DEPTH_TEST)
            glColor3f(1.0, 0.1, 0.1)
            glPointSize(12.0)
            glBegin(GL_POINTS)
            glVertex3f(px, py, pz)
            glEnd()

            # Küçük artı işareti
            s = 0.02
            glBegin(GL_LINES)
            glVertex3f(px - s, py,     pz); glVertex3f(px + s, py,     pz)
            glVertex3f(px,     py - s, pz); glVertex3f(px,     py + s, pz)
            glEnd()
            glEnable(GL_DEPTH_TEST)

    # ---- Etkileşim ----
    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            if event.buttons() & Qt.LeftButton:
                self.rotation_x += dy * 0.5
                self.rotation_y += dx * 0.5
            elif event.buttons() & Qt.RightButton:
                self.rotation_z += dx * 0.5
            self.last_mouse_pos = event.pos()
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        print("Zoom:", delta)
        self.zoom_distance += delta * 0.01
        self.zoom_distance = max(-50.0, min(-1.0, self.zoom_distance))
        self.update()

    # ---- Viewer API ----
    def set_pick_point(self, p3, normal=None, ray_dir=None):
        self._picked     = None if p3     is None else np.array(p3,     dtype=np.float64)
        self._picked_n   = None if normal is None else np.array(normal, dtype=np.float64)
        self._picked_ray = None if ray_dir is None else np.array(ray_dir, dtype=np.float64)
        self.update()

    def set_section_paths(self, paths):
        self._section_paths = paths or []
        self.update()

    # --- Ray üretimi (projeksiyon veya barycentrik pick için kullanılabilir) ---
    def compute_ray_from_screen(self, qpoint):
        MV = self._cached_model
        P  = self._cached_proj
        vp = self._cached_vp
        if MV is None or P is None or vp is None:
            return None, None

        dpr = float(self.devicePixelRatioF())

        # Ekran (Qt, üst-sol) -> pencere (OpenGL, alt-sol) piksel koordinatı
        wx = qpoint.x() * dpr
        wy = (float(vp[3]) - 1.0) - (qpoint.y() * dpr)

        # Window (piksel) -> NDC  (-1..+1)
        x_ndc = ((wx - float(vp[0])) / max(float(vp[2]), 1.0)) * 2.0 - 1.0
        y_ndc = ((wy - float(vp[1])) / max(float(vp[3]), 1.0)) * 2.0 - 1.0

        invP  = np.linalg.inv(P)
        invMV = np.linalg.inv(MV)

        near_clip = np.array([x_ndc, y_ndc, -1.0, 1.0], dtype=np.float64)
        far_clip  = np.array([x_ndc, y_ndc,  1.0, 1.0], dtype=np.float64)

        near_eye = invP @ near_clip
        far_eye  = invP @ far_clip
        near_eye = near_eye[:3] / max(abs(near_eye[3]), 1e-12)
        far_eye  = far_eye[:3]  / max(abs(far_eye[3]),  1e-12)

        dir_eye = far_eye - near_eye
        dir_eye /= max(np.linalg.norm(dir_eye), 1e-12)

        origin_obj = (invMV @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        dir_obj    = (invMV @ np.array([dir_eye[0], dir_eye[1], dir_eye[2], 0.0]))[:3]
        dir_obj   /= max(np.linalg.norm(dir_obj), 1e-12)

        return origin_obj, dir_obj

    def get_camera_forward(self):
        MV = self._cached_model
        if MV is None:
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
        invMV = np.linalg.inv(MV)
        f4 = invMV @ np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float64)
        f  = f4[:3]
        n  = np.linalg.norm(f)
        return f / n if n > 0 else np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def compute_ray_from_window_pixels(self, mx: float, my: float):
        """
        Window (bottom-left origin, piksel) koordinatından 3B ray üretir.
        Qt'dan gelen qpoint'e GEREK YOK; ok tuşu 'nişangâh' pikseli için ideal.
        Dönüş: (origin[3], direction[3]) veya (None, None)
        """
        MV = self._cached_model
        P = self._cached_proj
        vp = self._cached_vp
        if MV is None or P is None or vp is None:
            return None, None

        # Window (piksel) -> NDC  (-1..+1)
        x_ndc = ((float(mx) - float(vp[0])) / max(float(vp[2]), 1.0)) * 2.0 - 1.0
        y_ndc = ((float(my) - float(vp[1])) / max(float(vp[3]), 1.0)) * 2.0 - 1.0

        invP = np.linalg.inv(P)
        invMV = np.linalg.inv(MV)

        near_clip = np.array([x_ndc, y_ndc, -1.0, 1.0], dtype=np.float64)
        far_clip  = np.array([x_ndc, y_ndc,  1.0, 1.0], dtype=np.float64)

        near_eye = invP @ near_clip
        far_eye  = invP @ far_clip
        near_eye = near_eye[:3] / max(abs(near_eye[3]), 1e-12)
        far_eye  = far_eye[:3]  / max(abs(far_eye[3]),  1e-12)

        dir_eye = far_eye - near_eye
        n = np.linalg.norm(dir_eye)
        if n <= 0:
            return None, None
        dir_eye /= n

        origin_obj = (invMV @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        dir_obj    = (invMV @ np.array([dir_eye[0], dir_eye[1], dir_eye[2], 0.0]))[:3]
        m = np.linalg.norm(dir_obj)
        if m <= 0:
            return None, None
        dir_obj /= m
        return origin_obj, dir_obj
