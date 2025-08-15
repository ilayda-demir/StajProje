# workers/process_model_loader_worker.py
from PyQt5.QtCore import QThread, pyqtSignal
from multiprocessing import Process, Pipe
import numpy as np
import tempfile, os, traceback

def _child_load(file_path, conn):
    try:
        from adapters.model_loader import ModelLoader
        loader = ModelLoader()
        ok = loader.load_model(file_path)
        if not ok:
            conn.send({"ok": False, "err": f"Model yüklenemedi: {file_path}"})
            return

        V = loader.get_vertices()  # normalized
        F = loader.get_faces()

        tmp = tempfile.NamedTemporaryFile(prefix="mesh_", suffix=".npz", delete=False)
        tmp_path = tmp.name
        tmp.close()
        np.savez_compressed(tmp_path, V=V, F=F)

        conn.send({"ok": True, "path": tmp_path})
    except Exception as e:
        conn.send({"ok": False, "err": f"{e}\n{traceback.format_exc()}"})
    finally:
        try:
            conn.close()
        except Exception:
            pass


class ProcessModelLoaderWorker(QThread):
    loaded = pyqtSignal(object, object)  # (V, F)
    error  = pyqtSignal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self._proc = None
        self._parent_conn = None

    def cancel(self):
        """UI kapanırken veya kullanıcı iptal edince çağır."""
        try:
            self.requestInterruption()
        except Exception:
            pass
        try:
            if self._parent_conn is not None:
                self._parent_conn.close()   # run() içindeki poll/recv kırılır
        except Exception:
            pass
        try:
            if self._proc is not None and self._proc.is_alive():
                self._proc.terminate()
        except Exception:
            pass

    def run(self):
        parent_conn, child_conn = Pipe(duplex=False)
        self._parent_conn = parent_conn

        self._proc = Process(target=_child_load, args=(self.file_path, child_conn))
        self._proc.daemon = True
        self._proc.start()

        msg = None
        try:
            # Kısa aralıklarla bak: iptal isteği gelirse çık
            while True:
                if self.isInterruptionRequested():
                    break
                if parent_conn.poll(0.05):
                    msg = parent_conn.recv()
                    break
        except EOFError:
            msg = {"ok": False, "err": "İletişim kesildi (EOF)."}
        except Exception as e:
            msg = {"ok": False, "err": f"{e}\n{traceback.format_exc()}"}
        finally:
            try:
                parent_conn.close()
            except Exception:
                pass
            try:
                if self._proc is not None:
                    self._proc.join(timeout=0.1)
            except Exception:
                pass

        # İptal edilmişse hiçbir sinyal yaymadan sessizce bit
        if msg is None:
            return

        if not msg.get("ok"):
            self.error.emit(msg.get("err") or "Bilinmeyen hata")
            return

        path = msg["path"]
        try:
            with np.load(path) as data:
                V = data["V"]
                F = data["F"]
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

        self.loaded.emit(V, F)
