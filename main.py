import os
import sys

# Uyumluluk ve performans ayarları
os.environ["TRIMESH_NO_EMBREE"] = "1"   # trimesh hız/uyumluluk
# os.environ["QT_OPENGL"] = "software"  # OpenGL sorun olursa açılabilir

from PyQt5.QtCore import QCoreApplication, Qt
QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QSurfaceFormat

from adapters.ui_adapter import MainWindow

if __name__ == "__main__":
    # OpenGL format ayarları
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setVersion(2, 1)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(0)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
