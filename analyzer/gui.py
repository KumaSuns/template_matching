"""
analyzer: 動画の再生・コマ送り（保存・トリミングなし）
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSettings, QEvent
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QFrame,
    QLayout,
    QPushButton,
    QLabel,
    QSlider,
    QSplitter,
    QFileDialog,
    QComboBox,
    QMessageBox,
)

class ClickableSlider(QSlider):
    def _value_to_x(self, value):
        w = self.width()
        margin = 12
        span = max(1, w - 2 * margin)
        r = (value - self.minimum()) / max(1, self.maximum() - self.minimum())
        return margin + r * span

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        if self.orientation() != Qt.Horizontal:
            super().mousePressEvent(event)
            return
        pos_x = int(event.position().x())
        handle_center = self._value_to_x(self.value())
        if abs(pos_x - handle_center) <= 10:
            super().mousePressEvent(event)
            return
        w = self.width()
        margin = 12
        span = max(1, w - 2 * margin)
        val = self.minimum() + (self.maximum() - self.minimum()) * (pos_x - margin) / span
        self.setValue(int(max(self.minimum(), min(self.maximum(), val))))
        event.accept()

FRAME_STEP_OPTIONS = [
    ("−30", -30), ("−10", -10), ("−5", -5), ("−1", -1),
    ("+1", 1), ("+5", 5), ("+10", 10), ("+30", 30),
]

def cv2_to_qpixmap(bgr_image, scale_w: int = 0, scale_h: int = 0, fast: bool = False):
    if bgr_image is None or bgr_image.size == 0:
        return QPixmap()
    h, w = bgr_image.shape[:2]
    if len(bgr_image.shape) == 2:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    bytes_per_line = rgb.shape[2] * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg.copy())
    if scale_w > 0 and scale_h > 0 and (scale_w < w or scale_h < h):
        mode = Qt.FastTransformation if fast else Qt.SmoothTransformation
        pix = pix.scaled(scale_w, scale_h, Qt.KeepAspectRatio, mode)
    return pix


class PlaybackThread(QThread):
    frame_ready = Signal(int, object)
    finished_playback = Signal()

    def __init__(self, path: str, fps: float, total_frames: int, speed: float = 1.0):
        super().__init__()
        self.path = path
        self.fps = max(1.0, fps)
        self.total_frames = total_frames
        self.speed = max(0.25, min(1.0, speed))
        self.start_frame = 0
        self._stop = False

    def set_start_frame(self, frame_index: int):
        self.start_frame = max(0, min(frame_index, self.total_frames - 1))

    def stop(self):
        self._stop = True

    def run(self):
        self._stop = False
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            return
        try:
            frame_interval = 1.0 / (self.fps * self.speed)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            frame_index = self.start_frame
            while not self._stop and frame_index < self.total_frames:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                self.frame_ready.emit(frame_index, frame.copy())
                frame_index += 1
                if frame_index >= self.total_frames:
                    self.finished_playback.emit()
                    break
                elapsed = time.perf_counter() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
        finally:
            cap.release()


class AnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap: cv2.VideoCapture | None = None
        self.video_path: str | None = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.fps = 30.0
        self.playing = False
        self.playback_thread: PlaybackThread | None = None
        self._frame_width = 0
        self._frame_height = 0
        self._current_frame_np = None
        self._disp_pix_w = 0
        self._disp_pix_h = 0
        self._disp_off_x = 0
        self._disp_off_y = 0
        self._pending_playback_frame = None
        self._pending_playback_index = -1
        self._last_paint_time = 0.0
        _settings = QSettings("analyzer", "main")
        self._last_video_dir = _settings.value("lastVideoDir", "", type=str)
        self.setWindowTitle("analyzer")
        self.setMinimumSize(980, 1000)
        self.resize(1020, 1040)
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setSizeConstraint(QLayout.SetMinimumSize)
        main.setSpacing(10)
        main.setContentsMargins(8, 8, 8, 8)

        open_area = QFrame()
        open_area.setFrameStyle(QFrame.StyledPanel)
        open_area.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        open_area.setFixedHeight(48)
        open_area_layout = QHBoxLayout(open_area)
        open_area_layout.setContentsMargins(6, 6, 6, 6)
        self.btn_open = QPushButton("動画を開く")
        self.label_path = QLabel("（未選択）")
        self.label_path.setStyleSheet("color: #666;")
        open_area_layout.addWidget(self.btn_open)
        open_area_layout.addWidget(self.label_path, 1)
        main.addWidget(open_area)
        self.btn_open.clicked.connect(self.open_video)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        video_area = QFrame()
        video_area.setFrameStyle(QFrame.StyledPanel)
        video_area.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        video_area.setMinimumSize(560, 720)
        video_area.setFixedHeight(720)
        video_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vl = QVBoxLayout(video_area)
        vl.setContentsMargins(0, 0, 0, 0)
        self.label_video = QLabel()
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setStyleSheet("color: #999;")
        self.label_video.setText("動画を開いてください")
        self.label_video.installEventFilter(self)
        vl.addWidget(self.label_video)
        left_layout.addWidget(video_area)

        seek_area = QFrame()
        seek_area.setFrameStyle(QFrame.StyledPanel)
        seek_area.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        seek_area_layout = QHBoxLayout(seek_area)
        seek_area_layout.setContentsMargins(6, 6, 6, 6)
        self.label_time = QLabel("0:00 / 0:00")
        self.label_time.setMinimumWidth(90)
        seek_area_layout.addWidget(self.label_time)
        self.slider_seek = ClickableSlider(Qt.Horizontal)
        self.slider_seek.setMinimum(0)
        self.slider_seek.setMaximum(0)
        self.slider_seek.sliderMoved.connect(self.on_seek)
        self.slider_seek.valueChanged.connect(self.on_seek)
        self.slider_seek.installEventFilter(self)
        seek_area_layout.addWidget(self.slider_seek, 1)
        left_layout.addWidget(seek_area)

        ctrl_area = QFrame()
        ctrl_area.setFrameStyle(QFrame.StyledPanel)
        ctrl_area.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        ctrl_area_layout = QHBoxLayout(ctrl_area)
        ctrl_area_layout.setContentsMargins(6, 6, 6, 6)
        ctrl_area_layout.addStretch()
        self.btn_play = QPushButton("再生")
        self.btn_pause = QPushButton("一時停止")
        self.btn_stop = QPushButton("停止")
        self.combo_speed = QComboBox()
        for label, value in [("1x", 1.0), ("1/2x", 0.5), ("1/4x", 0.25)]:
            self.combo_speed.addItem(label, value)
        self.combo_speed.setCurrentIndex(0)
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        self.btn_play.clicked.connect(self.start_play)
        self.btn_pause.clicked.connect(self.pause_play)
        self.btn_stop.clicked.connect(self.stop_play)
        ctrl_area_layout.addWidget(self.btn_play)
        ctrl_area_layout.addWidget(self.btn_pause)
        ctrl_area_layout.addWidget(self.btn_stop)
        ctrl_area_layout.addWidget(QLabel("速度:"))
        ctrl_area_layout.addWidget(self.combo_speed)
        ctrl_area_layout.addStretch()
        left_layout.addWidget(ctrl_area)

        step_area = QFrame()
        step_area.setFrameStyle(QFrame.StyledPanel)
        step_area.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        step_area_layout = QHBoxLayout(step_area)
        step_area_layout.setContentsMargins(6, 6, 6, 6)
        step_area_layout.addStretch()
        self.step_buttons = []
        for label, delta in FRAME_STEP_OPTIONS:
            btn = QPushButton(label)
            btn.setFixedWidth(52)
            btn.clicked.connect(lambda checked=False, d=delta: self.do_frame_step(d))
            step_area_layout.addWidget(btn)
            self.step_buttons.append(btn)
        step_area_layout.addStretch()
        left_layout.addWidget(step_area)

        left_layout.addStretch()
        splitter.addWidget(left)

        center = QWidget()
        center.setMinimumWidth(80)
        center.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(center)

        right = QWidget()
        right.setMinimumWidth(80)
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(right)

        splitter.setSizes([400, 400, 400])
        main.addWidget(splitter)
        self.update_ui_state(False)

    def update_ui_state(self, has_video: bool):
        self.btn_play.setEnabled(has_video)
        self.btn_pause.setEnabled(has_video)
        self.btn_stop.setEnabled(has_video)
        self.combo_speed.setEnabled(has_video)
        self.slider_seek.setEnabled(has_video)
        for btn in self.step_buttons:
            btn.setEnabled(has_video)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "動画を開く", self._last_video_dir,
            "Video (*.mp4 *.avi *.mov *.mkv *.wmv *.webm);;All (*.*)"
        )
        if not path:
            return
        self._last_video_dir = str(Path(path).parent)
        QSettings("analyzer", "main").setValue("lastVideoDir", self._last_video_dir)
        self._close_capture()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "エラー", "動画を開けませんでした。")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_path = path
        self.current_frame_index = 0
        self._frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        p = Path(path)
        self.label_path.setText(p.name)
        self.label_path.setToolTip(str(p.resolve()))
        self.slider_seek.setMaximum(max(0, self.total_frames - 1))
        self.update_ui_state(True)
        self._show_frame_at(self.current_frame_index)
        self._update_time_label()

    def _close_capture(self):
        self._stop_playback_thread()
        if self.cap:
            self.playing = False
            self.cap.release()
            self.cap = None
        self.video_path = None
        self._current_frame_np = None
        self._frame_width = 0
        self._frame_height = 0
        self.update_ui_state(False)
        self.label_video.setText("動画を開いてください")
        self.label_video.setStyleSheet("color: #666;")

    def _frame_to_time(self, frame_index: int) -> str:
        if self.fps <= 0:
            return "0:00"
        sec = frame_index / self.fps
        return f"{int(sec // 60)}:{int(sec % 60):02d}"

    def _update_time_label(self):
        if self.cap is None:
            self.label_time.setText("0:00 / 0:00")
            return
        self.label_time.setText(f"{self._frame_to_time(self.current_frame_index)} / {self._frame_to_time(self.total_frames)}")
        self.slider_seek.blockSignals(True)
        self.slider_seek.setValue(self.current_frame_index)
        self.slider_seek.blockSignals(False)

    def _display_size(self):
        w = self.label_video.width()
        h = self.label_video.height()
        return (w, h) if w > 0 and h > 0 else (0, 0)

    def _show_frame_at(self, frame_index: int):
        if self.cap is None or self.total_frames == 0:
            return
        frame_index = max(0, min(frame_index, self.total_frames - 1))
        self.current_frame_index = frame_index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self._current_frame_np = frame.copy()
            self._paint_frame()
        self._update_time_label()

    def on_seek(self, value: int):
        self._show_frame_at(value)

    def _paint_frame(self, fast: bool = False):
        if self._current_frame_np is None:
            return
        sw, sh = self._display_size()
        pix = cv2_to_qpixmap(self._current_frame_np, sw, sh, fast=fast)
        self._disp_pix_w = pix.width()
        self._disp_pix_h = pix.height()
        self._disp_off_x = (self.label_video.width() - pix.width()) / 2
        self._disp_off_y = (self.label_video.height() - pix.height()) / 2
        self.label_video.setPixmap(pix)
        self.label_video.setStyleSheet("")

    def _stop_playback_thread(self):
        if self.playback_thread and self.playback_thread.isRunning():
            self.playback_thread.stop()
            self.playback_thread.wait(2000)
        self.playback_thread = None

    def _on_playback_frame(self, frame_index: int, frame: np.ndarray):
        self._pending_playback_index = frame_index
        self._pending_playback_frame = frame
        now = time.perf_counter()
        if self._last_paint_time == 0.0 or now - self._last_paint_time >= 0.032:
            self._flush_playback_frame()

    def _flush_playback_frame(self):
        if self._pending_playback_frame is None:
            return
        self._last_paint_time = time.perf_counter()
        frame_index = self._pending_playback_index
        frame = self._pending_playback_frame
        self._pending_playback_frame = None
        self._pending_playback_index = -1
        self.current_frame_index = frame_index
        self._current_frame_np = frame
        self._paint_frame(fast=True)
        self._update_time_label()

    def _on_playback_finished(self):
        self.playing = False
        self.playback_thread = None
        self._flush_playback_frame()
        self._update_time_label()

    def _on_speed_changed(self):
        if self.playing and self.playback_thread and self.playback_thread.isRunning():
            self._stop_playback_thread()
            self.start_play()

    def start_play(self):
        if self.cap is None or not self.video_path or self.total_frames == 0:
            return
        self._stop_playback_thread()
        self._pending_playback_frame = None
        self._last_paint_time = 0.0
        self.playing = True
        try:
            speed = float(self.combo_speed.currentData())
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(0.25, min(1.0, speed))
        self.playback_thread = PlaybackThread(
            self.video_path, self.fps, self.total_frames, speed
        )
        self.playback_thread.set_start_frame(self.current_frame_index)
        self.playback_thread.frame_ready.connect(self._on_playback_frame)
        self.playback_thread.finished_playback.connect(self._on_playback_finished)
        self.playback_thread.start()

    def pause_play(self):
        self.playing = False
        self._stop_playback_thread()

    def eventFilter(self, obj, event):
        slider = getattr(self, "slider_seek", None)
        if slider is not None and obj == slider and event.type() == QEvent.MouseButtonPress and self.playing:
            self.pause_play()
        return super().eventFilter(obj, event)

    def stop_play(self):
        self.pause_play()
        self._show_frame_at(0)

    def do_frame_step(self, delta: int):
        if self.cap is None:
            return
        next_index = self.current_frame_index + delta
        next_index = max(0, min(next_index, self.total_frames - 1))
        self._show_frame_at(next_index)

    def closeEvent(self, event):
        self._close_capture()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = AnalyzerWindow()
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
