# workers/process_model_loader_worker.py
from PyQt5.QtCore import QThread, pyqtSignal
from multiprocessing import Process, Pipe
import numpy as np
import tempfile, os, traceback
import os, tempfile

def _child_load(file_path, conn):
    try:
        from adapters.model_loader import ModelLoader
        loader = ModelLoader()
        ok = loader.load_model(file_path)
        if not ok:
            conn.send({"ok": False, "err": f"Model could not be loaded: {file_path}"})
            return

        V = loader.get_vertices()  # normalized
        F = loader.get_faces()
        C = loader.center
        S = loader.scale

        # Use a temp folder on D drive
        TMP_DIR = r"D:\temp"
        os.makedirs(TMP_DIR, exist_ok=True)

        tmp = tempfile.NamedTemporaryFile(prefix="mesh_", suffix=".npz", dir=TMP_DIR, delete=False)
        tmp_path = tmp.name
        tmp.close()
        np.savez_compressed(tmp_path, V=V, F=F, C=C, S=S)

        conn.send({"ok": True, "path": tmp_path})
    except Exception as e:
        conn.send({"ok": False, "err": f"{e}\n{traceback.format_exc()}"})
    finally:
        try:
            conn.close()
        except Exception:
            pass


class ProcessModelLoaderWorker(QThread):
    loaded = pyqtSignal(object, object, object, object)  # (V, F, C, S)
    error  = pyqtSignal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self._proc = None
        self._parent_conn = None

    def cancel(self):
        """Called when UI is closing or the user cancels."""
        try:
            self.requestInterruption()
        except Exception:
            pass
        try:
            if self._parent_conn is not None:
                self._parent_conn.close()   # breaks poll/recv inside run()
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
            # Check in short intervals: exit if cancellation requested
            while True:
                if self.isInterruptionRequested():
                    break
                if parent_conn.poll(0.05):
                    msg = parent_conn.recv()
                    break
        except EOFError:
            msg = {"ok": False, "err": "Communication lost (EOF)."}
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

        # If cancelled, exit silently without emitting any signals
        if msg is None:
            return

        if not msg.get("ok"):
            self.error.emit(msg.get("err") or "Unknown error")
            return

        path = msg["path"]
        try:
            with np.load(path) as data:
                V = data["V"]
                F = data["F"]
                C = data["C"]
                S = float(data["S"])
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

        self.loaded.emit(V, F, C, S)
