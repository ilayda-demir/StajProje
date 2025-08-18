# adapters/opengl_viewer.py
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QSurfaceFormat

from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluUnProject
import numpy as np


class OpenGLViewer(QOpenGLWidget):
    """
    Basit fixed-pipeline viewer:
      - Aydınlatmalı doldurma + tel-kafes overlay
      - Transparan mesh (alpha)
      - Picking için model/projection cache
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        fmt.setVersion(2, 1)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(0)
        self.setFormat(fmt)

        # Mesh verisi
        self.vertices = None
        self.faces = None
        self.normals_v = None

        # Kamera
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.zoom_distance = -6.0
        self.last_mouse_pos = None
        self.setFocusPolicy(Qt.StrongFocus)

        # Görünüm
        self.base_color = (0.70, 0.72, 0.76)
        self.mesh_alpha = 0.45           # varsayılan şeffaflık
        self.show_edges = True

        # Picking / section
        self._picked = None
        self._picked_n = None
        self._picked_ray = None
        self._section_paths = []

        # Picking için cache
        self._cached_model = None
        self._cached_proj = None
        self._cached_vp = None

    # ---------- Public API ----------
    def load_model_data(self, vertices, faces):
        V = np.asarray(vertices, dtype=np.float32)
        F = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)

        self.vertices = V
        self.faces = F
        self.normals_v = self._compute_vertex_normals(V, F)

        self.rotation_x = self.rotation_y = self.rotation_z = 0.0

        self.last_mouse_pos = None

        self._picked = None
        self._section_paths = []
        self.update()

    def set_section_paths(self, paths):
        self._section_paths = paths or []
        self.update()

    def set_pick_point(self, p3, normal=None, ray_dir=None):
        self._picked = None if p3 is None else np.array(p3, dtype=np.float64)
        self._picked_n = None if normal is None else np.array(normal, dtype=np.float64)
        self._picked_ray = None if ray_dir is None else np.array(ray_dir, dtype=np.float64)
        self.update()

    def set_mesh_alpha(self, a: float):
        """Üst bardaki kontrol ile şeffaflığı güncelle."""
        self.mesh_alpha = float(max(0.0, min(1.0, a)))
        self.update()

    # ---------- GL lifecycle ----------
    def initializeGL(self):
        glDisable(GL_MULTISAMPLE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glClearDepth(1.0)
        glClearColor(0.18, 0.18, 0.18, 1.0)

        # Aydınlatma
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.25, 0.25, 0.25, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  (0.90, 0.90, 0.90, 1.0))
        glLightfv(GL_LIGHT0, GL_POSITION, (3.0, 4.0, 6.0, 1.0))

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        # Z-fighting azalt
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # >>> Transparanlık
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # GLUT güvenli başlatma (eksende yazı için)
        try:
            from OpenGL.GLUT import glutInit
            glutInit()
        except Exception:
            pass

        # İlk projeksiyon
        w = max(1, self.width())
        h = max(1, self.height())
        self.resizeGL(w, h)

    def resizeGL(self, w, h):
        dpr = float(self.devicePixelRatioF())
        glViewport(0, 0, max(1, int(w * dpr)), max(1, int(h * dpr)))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = (w / h) if h else 1.0
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Kamera
        glTranslatef(0.0, 0.0, self.zoom_distance)
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        glRotatef(self.rotation_z, 0.0, 0.0, 1.0)

        # Cache matrisler
        model = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        vp = glGetIntegerv(GL_VIEWPORT)
        self._cached_model = np.array(model, dtype=np.float64).reshape((4, 4), order='F')
        self._cached_proj = np.array(proj, dtype=np.float64).reshape((4, 4), order='F')
        self._cached_vp = np.array(vp, dtype=np.int32)

        # Eksenler (+ etiketler)
        self._draw_axes()

        # Mesh
        if self.vertices is not None and self.faces is not None:
            V = self.vertices.astype(np.float32, copy=False)
            F = self.faces.astype(np.uint32, copy=False).ravel()
            N = self.normals_v.astype(np.float32, copy=False) if self.normals_v is not None else None

            # ---- Doldurma (transparan) ----
            glEnable(GL_LIGHTING)
            glColor4f(self.base_color[0], self.base_color[1], self.base_color[2], self.mesh_alpha)

            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, V)
            if N is not None:
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, N)

            glDrawElements(GL_TRIANGLES, F.size, GL_UNSIGNED_INT, F)

            if N is not None:
                glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)

            # ---- Tel-kafes overlay ----
            if self.show_edges:
                glDisable(GL_LIGHTING)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glLineWidth(1.2)
                glColor3f(0.08, 0.08, 0.08)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, V)
                glDrawElements(GL_TRIANGLES, F.size, GL_UNSIGNED_INT, F)
                glDisableClientState(GL_VERTEX_ARRAY)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Kesit polylineleri
        if self._section_paths:
            glDisable(GL_LIGHTING)
            glColor3f(1.0, 1.0, 0.0)
            for poly in self._section_paths:
                if poly is None:
                    continue
                P = np.asarray(poly, dtype=np.float32)
                if P.ndim != 2 or P.shape[0] < 2 or P.shape[1] != 3:
                    continue
                if not np.all(np.isfinite(P)):
                    continue
                glBegin(GL_LINE_STRIP)
                for p in P:
                    glVertex3f(float(p[0]), float(p[1]), float(p[2]))
                glEnd()

        # Seçili nokta (kırmızı artı)
        if self._picked is not None:
            px, py, pz = map(float, self._picked)
            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)
            glColor3f(1.0, 0.7, 0.1)

            glPointSize(12.0)
            glBegin(GL_POINTS)
            glVertex3f(px, py, pz)
            glEnd()
            s = 0.02
            glBegin(GL_LINES)
            glVertex3f(px - s, py, pz); glVertex3f(px + s, py, pz)
            glVertex3f(px, py - s, pz); glVertex3f(px, py + s, pz)
            glEnd()
            glEnable(GL_DEPTH_TEST)

    # ---------- Interaction ----------
    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()
        super().mousePressEvent(event)
        self.update()

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
        else:
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom_distance += delta * 0.01
        self.zoom_distance = max(-50.0, min(-1.0, self.zoom_distance))
        self.update()

    # ---------- Picking ----------
    def pick_point_from_qt(self, qpoint):
        MV = self._cached_model
        P = self._cached_proj
        vp = self._cached_vp
        if MV is None or P is None or vp is None:
            return None

        dpr = float(self.devicePixelRatioF())
        wx = qpoint.x() * dpr
        wy = (float(vp[3]) - 1.0) - (qpoint.y() * dpr)

        self.makeCurrent()
        try:
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            ix, iy = int(round(wx)), int(round(wy))
            depth = glReadPixels(ix, iy, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
            if depth is None:
                return None
            z = float(depth[0][0]) if hasattr(depth, "__getitem__") else float(depth)
            if z >= 1.0 - 1e-6:
                return None
            x, y, z = gluUnProject(wx, wy, z, MV, P, vp)
            return np.array([x, y, z], dtype=np.float64)
        finally:
            self.doneCurrent()

    @staticmethod
    def _compute_vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
        N = np.zeros_like(V, dtype=np.float32)
        v0 = V[F[:, 0]]; v1 = V[F[:, 1]]; v2 = V[F[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for i in range(3):
            np.add.at(N, F[:, i], fn)
        lens = np.linalg.norm(N, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        return (N / lens).astype(np.float32)

    def _draw_axes(self):
        # Eksen çizgileri
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X (kırmızı)
        glColor3f(1.0, 0.0, 0.0); glVertex3f(-2.0, 0.0, 0.0); glVertex3f(2.0, 0.0, 0.0)
        # Y (yeşil)
        glColor3f(0.0, 1.0, 0.0); glVertex3f(0.0, -2.0, 0.0); glVertex3f(0.0, 2.0, 0.0)
        # Z (mavi)
        glColor3f(0.0, 0.0, 1.0); glVertex3f(0.0, 0.0, -2.0); glVertex3f(0.0, 0.0, 2.0)
        glEnd()

        # Uçlara eksen etiketleri (eksene uygun renkler)
        self._draw_text_3d(2.15, 0.0, 0.0, "X", (1.0, 0.0, 0.0))
        self._draw_text_3d(0.0, 2.15, 0.0, "Y", (0.0, 1.0, 0.0))
        self._draw_text_3d(0.0, 0.0, 2.15, "Z", (0.0, 0.0, 1.0))

        glEnable(GL_LIGHTING)

    def compute_ray_from_window_pixels(self, mx, my):
        MV = self._cached_model
        P = self._cached_proj
        vp = self._cached_vp
        if MV is None or P is None or vp is None:
            return None, None
        # normalize window coords
        wx, wy = float(mx), float(my)
        # near ve far noktalarını dünya uzayına aç
        near = gluUnProject(wx, wy, 0.0, MV, P, vp)
        far = gluUnProject(wx, wy, 1.0, MV, P, vp)
        if near is None or far is None:
            return None, None
        o = np.array(near, dtype=np.float64)
        d = np.array(far, dtype=np.float64) - o
        n = np.linalg.norm(d)
        if n == 0:
            return None, None
        return o, d / n

    # ---------- Helpers ----------
    def _draw_text_3d(self, x, y, z, text, color=(1.0, 1.0, 1.0)):
        """3B uzayda tek satır yazı (eksene uygun renkte, state izolasyonlu)."""
        try:
            from OpenGL.GLUT import glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
        except Exception:
            return  # GLUT yoksa sessizce atla

        # Mevcut enable/disable bayraklarını koru
        glPushAttrib(GL_ENABLE_BIT)
        try:
            glDisable(GL_LIGHTING)     # yazı için düz renk
            glDisable(GL_DEPTH_TEST)   # her zaman görünür olsun (istersen bunu kaldırabilirsin)
            glColor3f(*color)
            glRasterPos3f(x, y, z)

            for ch in text:
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))
        finally:
            glPopAttrib()
