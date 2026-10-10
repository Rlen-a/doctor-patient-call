#!/usr/bin/env python3
"""
Модуль определения лица в кадре.

Используется клиентами врача и пациента: анализирует видео собеседника и
  * рисует зелёную рамку вокруг найденного лица;
  * пишет в углу кадра "FACE: YES" / "FACE: NO" (OpenCV не умеет кириллицу);
  * печатает в терминал по-русски, когда лицо появилось или пропало;
  * при желании (--voice) произносит это вслух (нужен pip install pyttsx3).

Как это работает:
  * Анализ идёт в ОТДЕЛЬНОМ потоке и берёт только самый свежий кадр, поэтому
    видео не тормозит, а ошибки детектора не ломают показ.
  * Основной детектор — YuNet (нейросеть из OpenCV, работает при плохом свете и
    с поворотом головы). Файл модели (~230 КБ) один раз скачивается рядом со
    скриптом. Если скачать не удалось (нет интернета) или версия OpenCV старая,
    автоматически используются каскады Хаара (альтернатива).
  * Состояние меняется только после нескольких одинаковых результатов подряд.

Проверка на своей камере:  python face_check.py
"""
import os
import queue
import threading
import time
import traceback
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
YUNET_FILE = os.path.join(HERE, "face_detection_yunet_2023mar.onnx")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


# ------------------------------------------------------------------ детекторы

class YuNetDetector:
    name = "YuNet"

    def __init__(self, path: str):
        self.det = cv2.FaceDetectorYN.create(path, "", (320, 320), 0.6, 0.3, 5000)

    def detect(self, img):
        h, w = img.shape[:2]
        scale = 640.0 / w if w > 640 else 1.0
        small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale != 1.0 else img
        sh, sw = small.shape[:2]
        self.det.setInputSize((sw, sh))
        _, faces = self.det.detect(small)
        out = []
        if faces is not None:
            for f in faces:
                x, y, fw, fh = f[:4]
                out.append((int(x / scale), int(y / scale), int(fw / scale), int(fh / scale)))
        return out


class HaarDetector:
    name = "Haar"

    def __init__(self):
        self.cascades = []
        for kind, fname in (("front", "haarcascade_frontalface_alt2.xml"),
                            ("front", "haarcascade_frontalface_default.xml"),
                            ("profile", "haarcascade_profileface.xml")):
            c = cv2.CascadeClassifier(cv2.data.haarcascades + fname)
            if not c.empty():
                self.cascades.append((kind, c))
        if not self.cascades:
            raise RuntimeError("Не найдены каскады Хаара в " + cv2.data.haarcascades)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, img):
        h, w = img.shape[:2]
        scale = 480.0 / w if w > 480 else 1.0
        small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale != 1.0 else img
        gray = self.clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        sw = gray.shape[1]
        m = max(24, int(sw * 0.06))
        for kind, casc in self.cascades:
            boxes = [tuple(b) for b in casc.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(m, m))]
            if kind == "profile":
                # профиль ищем в обе стороны: в зеркальном кадре и в обычном
                for (x, y, fw, fh) in casc.detectMultiScale(
                        cv2.flip(gray, 1), scaleFactor=1.1, minNeighbors=4, minSize=(m, m)):
                    boxes.append((sw - x - fw, y, fw, fh))
            if boxes:
                return [(int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))
                        for (x, y, fw, fh) in boxes]
        return []


def make_detector():
    """YuNet, если получится; иначе каскады Хаара."""
    if hasattr(cv2, "FaceDetectorYN"):
        try:
            if not os.path.exists(YUNET_FILE):
                print("Скачиваю модель детектора лиц YuNet (один раз)...")
                req = urllib.request.urlopen(YUNET_URL, timeout=10)
                data = req.read()
                with open(YUNET_FILE, "wb") as f:
                    f.write(data)
            return YuNetDetector(YUNET_FILE)
        except Exception as e:  # noqa: BLE001
            print(f"YuNet недоступен ({e}); использую каскады Хаара.")
            try:
                if os.path.exists(YUNET_FILE) and os.path.getsize(YUNET_FILE) < 10000:
                    os.remove(YUNET_FILE)  # битый файл
            except OSError:
                pass
    else:
        print("В этой версии OpenCV нет YuNet; использую каскады Хаара.")
    return HaarDetector()


# ------------------------------------------------------------------ озвучка

class Announcer:
    """Озвучивание фраз через pyttsx3 в отдельном потоке."""

    def __init__(self):
        self.ok = False
        try:
            import pyttsx3  # noqa: F401
        except ImportError:
            print("Голос отключён: установите pyttsx3 (pip install pyttsx3)")
            return
        self.ok = True
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        import pyttsx3
        try:
            engine = pyttsx3.init()
            for v in engine.getProperty("voices"):
                s = (str(v.id) + str(v.name)).lower()
                if any(k in s for k in ("russian", "ru-ru", "ru_ru", "irina", "pavel")):
                    engine.setProperty("voice", v.id)
                    break
        except Exception as e:  # noqa: BLE001
            print("Не удалось запустить синтез речи:", e)
            return
        while True:
            text = self.q.get()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:  # noqa: BLE001
                pass

    def say(self, text: str):
        if self.ok:
            self.q.put(text)


# ------------------------------------------------------------------ монитор

class FaceMonitor:
    def __init__(self, label: str = "Собеседник", every: int = 3,
                 confirm: int = 3, voice: bool = False):
        """
        label   — как называть человека в сообщениях ("Пациент", "Врач");
        every   — отправлять на анализ каждый N-й кадр;
        confirm — сколько одинаковых результатов подряд нужно для смены состояния;
        voice   — озвучивать смену состояния.
        """
        self.detector = make_detector()
        print(f"Детектор лиц: {self.detector.name}")
        self.label = label
        self.every = max(1, every)
        self.confirm = max(1, confirm)
        self.announcer = Announcer() if voice else None

        self.faces = []
        self.state = None  # None — ещё не определено, True — лицо есть, False — нет
        self.analyzed = 0
        self.last_ms = 0.0
        self.error = ""
        self._n = 0
        self._streak = 0
        self._lock = threading.Lock()
        self._frame = None
        self._event = threading.Event()
        self._alive = True
        threading.Thread(target=self._worker, daemon=True).start()

    # --- вызывается из потока видео

    def due(self) -> bool:
        """Нужно ли отправить текущий кадр на анализ."""
        self._n += 1
        return self._n % self.every == 0

    def submit(self, img) -> None:
        """Передаёт кадр на анализ (копия, не блокирует)."""
        with self._lock:
            self._frame = img.copy()
        self._event.set()

    def close(self) -> None:
        self._alive = False
        self._event.set()

    # --- рабочий поток

    def _worker(self):
        while self._alive:
            self._event.wait()
            self._event.clear()
            with self._lock:
                img, self._frame = self._frame, None
            if img is None:
                continue
            try:
                t0 = time.time()
                faces = self.detector.detect(img)
                self.last_ms = (time.time() - t0) * 1000
                self.faces = faces
                self.analyzed += 1
                self.error = ""
                self._vote(len(faces) > 0)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                if msg != self.error:
                    print("Ошибка определения лица:")
                    traceback.print_exc()
                self.error = msg

    def _vote(self, present: bool) -> None:
        if present == self.state:
            self._streak = 0
            return
        self._streak += 1
        if self._streak >= self.confirm:
            self.state = present
            self._streak = 0
            self._announce(present)

    def _announce(self, present: bool) -> None:
        stamp = time.strftime("%H:%M:%S")
        text = f"{self.label}: лицо в кадре" if present else f"{self.label}: лица в кадре нет"
        print(f"[{stamp}] {text}")
        if self.announcer:
            self.announcer.say(text)

    # --- диагностика и рисование

    def summary(self) -> str:
        st = {True: "лицо есть", False: "лица нет", None: "не определено"}[self.state]
        s = (f"[лицо] {self.detector.name}: проанализировано кадров {self.analyzed}, "
             f"{self.last_ms:.0f} мс/кадр, найдено лиц {len(self.faces)} -> {st}")
        if self.error:
            s += f"  <-- ОШИБКА: {self.error}"
        return s

    def draw(self, img) -> None:
        """Рисует рамки и статус прямо на кадре (img должен быть записываемым)."""
        for (x, y, w, h) in list(self.faces):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if self.state is True:
            text, color = "FACE: YES", (0, 200, 0)
        elif self.state is False:
            text, color = "FACE: NO", (0, 0, 255)
        else:
            text, color = "FACE: ...", (200, 200, 200)
        cv2.rectangle(img, (0, 0), (190, 32), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


if __name__ == "__main__":
    # Самопроверка на локальной камере. Выход — клавиша 'q'.
    cap = cv2.VideoCapture(0)
    mon = FaceMonitor(label="Вы", every=2)
    last = time.time()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if mon.due():
            mon.submit(frame)
        mon.draw(frame)
        cv2.imshow("face_check", frame)
        if time.time() - last > 3:
            print(mon.summary())
            last = time.time()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    mon.close()
    cap.release()
    cv2.destroyAllWindows()
