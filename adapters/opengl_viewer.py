# adapters/opengl_viewer.py
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QSurfaceFormat

from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluUnProject
import numpy as np


class OpenGLViewer(QOpenGLWidget):
    """
    Fixed-pipeline OpenGL viewer with:
      - proper lighting (per-vertex normals)
      - wireframe overlay so triangle edges are visible
      - cached model/projection matrices for accurate picking
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # Robust GL format (compat profile 2.1 for PyOpenGL fixed pipeline)
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        fmt.setVersion(2, 1)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(0)  # keep MSAA off; some Intel drivers are happier
        self.setFormat(fmt)

        # Mesh data
        self.vertices = None             # (N,3) float
        self.faces = None                # (M,3) uint / int
        self.normals_v = None            # (N,3) float per-vertex normals

        # View state
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.zoom_distance = -5.0
        self.last_mouse_pos = None
        self.setFocusPolicy(Qt.StrongFocus)

        # Appearance
        self.base_color = (0.70, 0.72, 0.76)  # neutral gray by default
        self.show_edges = True

        # Picking / section rendering
        self._picked = None
        self._picked_n = None
        self._picked_ray = None
        self._section_paths = []

        # Cached matrices for picking
        self._cached_model = None  # 4x4
        self._cached_proj = None   # 4x4
        self._cached_vp = None     # [x,y,w,h]

    # ---------- Public API ----------
    def load_model_data(self, vertices, faces):
        """Load normalized vertices/faces (no re-normalize here)."""
        V = np.asarray(vertices, dtype=np.float32)
        F = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)

        self.vertices = V
        self.faces = F
        self.normals_v = self._compute_vertex_normals(V, F)

        # reset camera
        self.rotation_x = self.rotation_y = self.rotation_z = 0.0
        self.zoom_distance = -5.0
        self.last_mouse_pos = None

        # clear picks/sections
        self._picked = None
        self._section_paths = []
        self.update()

    # ---------- GL lifecycle ----------
    def initializeGL(self):
        glDisable(GL_MULTISAMPLE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glClearDepth(1.0)
        glClearColor(0.12, 0.12, 0.12, 1.0)

        # Lighting (so surfaces are shaded, not flat neon)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.25, 0.25, 0.25, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  (0.90, 0.90, 0.90, 1.0))
        glLightfv(GL_LIGHT0, GL_POSITION, (3.0, 4.0, 6.0, 1.0))

        glEnable(GL_COLOR_MATERIAL)  # let glColor drive material
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        # Avoid z-fighting when overlaying wireframe on top of filled polys
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

        # Nice wireframe lines
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        try:
            ver = glGetString(GL_VERSION)
            ren = glGetString(GL_RENDERER)
            print("[GL] Version:", ver.decode() if ver else ver, "| Renderer:", ren.decode() if ren else ren)
        except Exception:
            pass

        # ensure a valid projection on first frame
        w = max(1, self.width())
        h = max(1, self.height())
        self.resizeGL(w, h)

    def resizeGL(self, w, h):
        dpr = float(self.devicePixelRatioF())
        vp_w = max(1, int(w * dpr))
        vp_h = max(1, int(h * dpr))
        glViewport(0, 0, vp_w, vp_h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = (w / h) if h else 1.0
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera
        glTranslatef(0.0, 0.0, self.zoom_distance)
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        glRotatef(self.rotation_z, 0.0, 0.0, 1.0)

        # Cache matrices for picking
        model = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        vp = glGetIntegerv(GL_VIEWPORT)
        self._cached_model = np.array(model, dtype=np.float64).reshape((4, 4), order='F')
        self._cached_proj = np.array(proj, dtype=np.float64).reshape((4, 4), order='F')
        self._cached_vp = np.array(vp, dtype=np.int32)

        # Axes
        self._draw_axes()

        # Mesh
        if self.vertices is not None and self.faces is not None:
            # ----- Filled pass (with lighting) -----
            glEnable(GL_LIGHTING)
            glColor3f(*self.base_color)

            V = self.vertices.astype(np.float32, copy=False)
            F = self.faces.astype(np.uint32, copy=False).ravel()
            N = None
            if self.normals_v is not None:
                N = self.normals_v.astype(np.float32, copy=False)

            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, V)

            if N is not None:
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, N)

            glDrawElements(GL_TRIANGLES, F.size, GL_UNSIGNED_INT, F)

            if N is not None:
                glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)

            # ----- Wireframe overlay -----
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

        # Section polylines
        if self._section_paths:
            glDisable(GL_LIGHTING)
            glColor3f(1.0, 1.0, 0.0)
            for poly in self._section_paths:
                if poly is None or len(poly) == 0:
                    continue
                glBegin(GL_LINE_STRIP)
                for p in poly:
                    px, py, pz = map(float, p)
                    glVertex3f(px, py, pz)
                glEnd()

        # Picked point (always visible)
        if self._picked is not None:
            px, py, pz = map(float, self._picked)
            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)
            glColor3f(1.0, 0.1, 0.1)
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
        self.zoom_distance += delta * 0.01
        self.zoom_distance = max(-50.0, min(-1.0, self.zoom_distance))
        self.update()

    # ---------- Viewer API ----------
    def set_pick_point(self, p3, normal=None, ray_dir=None):
        self._picked = None if p3 is None else np.array(p3, dtype=np.float64)
        self._picked_n = None if normal is None else np.array(normal, dtype=np.float64)
        self._picked_ray = None if ray_dir is None else np.array(ray_dir, dtype=np.float64)
        self.update()

    def set_section_paths(self, paths):
        self._section_paths = paths or []
        self.update()

    # ---------- Picking helpers ----------
    def compute_ray_from_screen(self, qpoint):
        MV = self._cached_model
        P = self._cached_proj
        vp = self._cached_vp
        if MV is None or P is None or vp is None:
            return None, None

        dpr = float(self.devicePixelRatioF())
        wx = qpoint.x() * dpr
        wy = (float(vp[3]) - 1.0) - (qpoint.y() * dpr)

        x_ndc = ((wx - float(vp[0])) / max(float(vp[2]), 1.0)) * 2.0 - 1.0
        y_ndc = ((wy - float(vp[1])) / max(float(vp[3]), 1.0)) * 2.0 - 1.0

        invP = np.linalg.inv(P)
        invMV = np.linalg.inv(MV)

        near_clip = np.array([x_ndc, y_ndc, -1.0, 1.0], dtype=np.float64)
        far_clip = np.array([x_ndc, y_ndc, 1.0, 1.0], dtype=np.float64)

        near_eye = invP @ near_clip
        far_eye = invP @ far_clip
        near_eye = near_eye[:3] / max(abs(near_eye[3]), 1e-12)
        far_eye = far_eye[:3] / max(abs(far_eye[3]), 1e-12)

        dir_eye = far_eye - near_eye
        dir_eye /= max(np.linalg.norm(dir_eye), 1e-12)

        origin_obj = (invMV @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        dir_obj = (invMV @ np.array([dir_eye[0], dir_eye[1], dir_eye[2], 0.0]))[:3]
        dir_obj /= max(np.linalg.norm(dir_obj), 1e-12)

        return origin_obj, dir_obj

    def get_camera_forward(self):
        MV = self._cached_model
        if MV is None:
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
        invMV = np.linalg.inv(MV)
        f4 = invMV @ np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float64)
        f = f4[:3]
        n = np.linalg.norm(f)
        return f / n if n > 0 else np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def compute_ray_from_window_pixels(self, mx: float, my: float):
        MV = self._cached_model
        P = self._cached_proj
        vp = self._cached_vp
        if MV is None or P is None or vp is None:
            return None, None

        x_ndc = ((float(mx) - float(vp[0])) / max(float(vp[2]), 1.0)) * 2.0 - 1.0
        y_ndc = ((float(my) - float(vp[1])) / max(float(vp[3]), 1.0)) * 2.0 - 1.0

        invP = np.linalg.inv(P)
        invMV = np.linalg.inv(MV)

        near_clip = np.array([x_ndc, y_ndc, -1.0, 1.0], dtype=np.float64)
        far_clip = np.array([x_ndc, y_ndc, 1.0, 1.0], dtype=np.float64)

        near_eye = invP @ near_clip
        far_eye = invP @ far_clip
        near_eye = near_eye[:3] / max(abs(near_eye[3]), 1e-12)
        far_eye = far_eye[:3] / max(abs(far_eye[3]), 1e-12)

        dir_eye = far_eye - near_eye
        n = np.linalg.norm(dir_eye)
        if n <= 0:
            return None, None
        dir_eye /= n

        origin_obj = (invMV @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        dir_obj = (invMV @ np.array([dir_eye[0], dir_eye[1], dir_eye[2], 0.0]))[:3]
        m = np.linalg.norm(dir_obj)
        if m <= 0:
            return None, None
        dir_obj /= m
        return origin_obj, dir_obj

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

    # ---------- Helpers ----------
    @staticmethod
    def _compute_vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
        """Average adjacent face normals to get smooth per-vertex normals."""
        N = np.zeros_like(V, dtype=np.float32)
        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        # accumulate to vertices
        for i in range(3):
            np.add.at(N, F[:, i], fn)
        # normalize
        lens = np.linalg.norm(N, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        N = N / lens
        return N.astype(np.float32)

    def _draw_axes(self):
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X (red)
        glColor3f(1.0, 0.0, 0.0); glVertex3f(-2.0, 0.0, 0.0); glVertex3f(2.0, 0.0, 0.0)
        # Y (green)
        glColor3f(0.0, 1.0, 0.0); glVertex3f(0.0, -2.0, 0.0); glVertex3f(0.0, 2.0, 0.0)
        # Z (blue)
        glColor3f(0.0, 0.0, 1.0); glVertex3f(0.0, 0.0, -2.0); glVertex3f(0.0, 0.0, 2.0)
        glEnd()
        glEnable(GL_LIGHTING)
