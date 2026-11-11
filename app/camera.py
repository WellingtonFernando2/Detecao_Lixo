import cv2
import numpy as np

CALIBRATION_FRAMES = 20
BLACK_THRESHOLD = 20


ALIAS = {"celular": "Iriun", "iriun": "Iriun"}


def tipo_camera(valor: str):
    return int(valor) if valor.isdigit() else valor


def procurar_indice(nome: str) -> int:
    from cv2_enumerate_cameras import enumerate_cameras

    nome = ALIAS.get(nome.lower(), nome)
    cameras = list(enumerate_cameras(cv2.CAP_DSHOW))
    for info in cameras:
        if nome.lower() in info.name.lower():
            return info.index
    nomes = ", ".join(f"[{c.index}] {c.name}" for c in cameras) or "nenhuma"
    raise RuntimeError(f"Camera '{nome}' nao encontrada. Cameras disponiveis: {nomes}. "
                       "Se for o celular, abra o Iriun no celular e no notebook.")


class Camera:
    def __init__(self, index, width: int = 1280, height: int = 720, recortar: bool = True):
        if isinstance(index, str):
            index = procurar_indice(index)
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Nao foi possivel abrir a camera. Verifique o --camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.crop = self._calibrate() if recortar else None
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {index}: {w}x{h}, recorte de faixas pretas: {self.crop}")

    def _calibrate(self):
        peak = None
        for _ in range(CALIBRATION_FRAMES):
            ok, frame = self.cap.read()
            if not ok:
                continue
            gray = frame.max(axis=2)
            peak = gray if peak is None else np.maximum(peak, gray)
        if peak is None:
            return None

        rows = np.where(peak.max(axis=1) > BLACK_THRESHOLD)[0]
        cols = np.where(peak.max(axis=0) > BLACK_THRESHOLD)[0]
        if len(rows) == 0 or len(cols) == 0:
            return None
        y1, y2, x1, x2 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        h, w = peak.shape
        if (y2 - y1) * (x2 - x1) > 0.95 * h * w:
            return None
        if (y2 - y1) < 0.4 * h or (x2 - x1) < 0.4 * w:
            return None
        return y1, y2, x1, x2

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        if self.crop:
            y1, y2, x1, x2 = self.crop
            frame = frame[y1:y2, x1:x2]
        return frame

    def release(self):
        self.cap.release()
