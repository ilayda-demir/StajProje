from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QSurfaceFormat
from OpenGL.GL import *
import numpy as np


class OpenGLViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertices = None
        self.faces = None

    def load_model_data(self, vertices, faces):
        # Normalize: merkeze getir ve uygun boyuta ölçekle
        vertices = np.array(vertices)
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        center = (min_coords + max_coords) / 2
        scale = np.linalg.norm(max_coords - min_coords)

        # Taşı ve ölçekle
        normalized = (vertices - center) / scale * 2.0

        self.vertices = normalized
        self.faces = faces
        self.update()

        self.last_mouse_pos = None
        self.rotation_x = 0.0
        self.rotation_y = 0.0

    def initializeGL(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h != 0 else 1
        glOrtho(-1 * aspect, 1 * aspect, -1, 1, -10, 10)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)

        if self.vertices is not None and self.faces is not None:
            glColor3f(0.0, 0.6, 1.0)  # Model rengi: mavi
            glBegin(GL_TRIANGLES)
            for face in self.faces:
                for idx in face:
                    if idx < len(self.vertices):
                        glVertex3fv(self.vertices[idx])
            glEnd()

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            self.rotation_x += dy * 0.5
            self.rotation_y += dx * 0.5
            self.last_mouse_pos = event.pos()
            self.update()


