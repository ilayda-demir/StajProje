from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective
import numpy as np


class OpenGLViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertices = None
        self.faces = None

        # Dönüş ve zoom değişkenleri
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.zoom_distance = -5.0  # ⬅️ Burada tanımlanmalı

        self.last_mouse_pos = None

        self.setFocusPolicy(Qt.StrongFocus)  # ⬅️ Wheel event için odak almalı

    def load_model_data(self, vertices, faces):
        # Normalize: merkeze getir ve uygun boyuta ölçekle
        vertices = np.array(vertices)
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        center = (min_coords + max_coords) / 2
        scale = np.linalg.norm(max_coords - min_coords)
        if scale == 0:
            scale = 1.0

        normalized = (vertices - center) / scale * 2.0

        self.vertices = normalized
        self.faces = faces
        self.update()

        # Sıfırla
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.zoom_distance = -5.0
        self.last_mouse_pos = None

    def initializeGL(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h != 0 else 1
        gluPerspective(45.0, aspect, 0.1, 100.0)  # Perspektif projeksiyon
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Zoom (kamera uzaklığı)
        glTranslatef(0.0, 0.0, self.zoom_distance)

        # Döndürmeler
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        glRotatef(self.rotation_z, 0.0, 0.0, 1.0)

        if self.vertices is not None and self.faces is not None:
            glColor3f(0.0, 0.6, 1.0)
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

            if event.buttons() & Qt.LeftButton:
                self.rotation_x += dy * 0.5
                self.rotation_y += dx * 0.5

            elif event.buttons() & Qt.RightButton:
                self.rotation_z += dx * 0.5

            self.last_mouse_pos = event.pos()
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        print("Zoom:", delta)  # Debug için
        self.zoom_distance += delta * 0.01  # ⬅️ Daha etkili zoom
        self.zoom_distance = max(-50.0, min(-1.0, self.zoom_distance))
        self.update()
