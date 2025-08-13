from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog
from adapters.opengl_viewer import OpenGLViewer
from adapters.model_loader import ModelLoader


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model Görüntüleyici")
        self.setGeometry(100, 100, 800, 600)

        self.viewer = OpenGLViewer()
        self.loader = ModelLoader()

        self.load_button = QPushButton("3D Model Yükle")
        self.load_button.clicked.connect(self.load_model)

        layout = QVBoxLayout()
        layout.addWidget(self.viewer)
        layout.addWidget(self.load_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "3D Model Seç", "", "3D Modeller (*.obj *.stl)")
        if file_path:
            if self.loader.load_model(file_path):
                vertices = self.loader.get_vertices()
                faces = self.loader.get_faces()
                self.viewer.load_model_data(vertices, faces)
