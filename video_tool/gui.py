"""
動画ツール: 取り込み・再生/一時停止・コマ送り・キャプチャ・トリミング
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QFrame,
    QPushButton,
    QLabel,
    QSlider,
    QFileDialog,
    QGroupBox,
    QSpinBox,
    QMessageBox,
    QScrollArea,
)

# コマ送り: 左＝前へ、右＝次へ。表示はシンプルに ±フレーム数
FRAME_STEP_OPTIONS = [
    ("−30", -30),
    ("−10", -10),
    ("−5", -5),
    ("−1", -1),
    ("+1", 1),
    ("+5", 5),
    ("+10", 10),
    ("+30", 30),
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
    """再生用: 別スレッドでフレームを読んでメインに渡す"""
    frame_ready = Signal(int, object)  # frame_index, frame (numpy copy)
    finished_playback = Signal()

    def __init__(self, path: str, fps: float, total_frames: int):
        super().__init__()
        self.path = path
        self.fps = max(1.0, fps)
        self.total_frames = total_frames
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
            frame_interval = 1.0 / self.fps
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


class VideoToolWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap: cv2.VideoCapture | None = None
        self.video_path: str | None = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.fps = 30.0
        self.playing = False
        self.playback_thread: PlaybackThread | None = None
        self.in_point = 0
        self.out_point = 0
        _settings = QSettings("template_matching", "video_tool")
        self._last_video_dir = _settings.value("lastVideoDir", "", type=str)
        self.setWindowTitle("動画ツール")
        self.setMinimumSize(1000, 720)
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ファイル
        file_group = QGroupBox("ファイル")
        file_layout = QHBoxLayout(file_group)
        self.btn_open = QPushButton("動画を開く")
        self.label_path = QLabel("（未選択）")
        self.label_path.setStyleSheet("color: #666;")
        file_layout.addWidget(self.btn_open)
        file_layout.addWidget(self.label_path)
        file_layout.addStretch()
        layout.addWidget(file_group)
        self.btn_open.clicked.connect(self.open_video)

        # 映像表示（高さを固定し、下のUIが必ず見えるようにする）
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.StyledPanel)
        video_frame.setFixedHeight(400)
        video_frame.setMinimumWidth(640)
        video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.label_video = QLabel()
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setMinimumSize(320, 200)
        self.label_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_video.setStyleSheet("background: #1a1a1a; color: #666; font-size: 14px;")
        self.label_video.setText("動画を開いてください")
        video_layout.addWidget(self.label_video)
        layout.addWidget(video_frame, 0, Qt.AlignTop)

        # 以下はスクロール可能（動画と被らない）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        # 再生コントロール
        ctrl_group = QGroupBox("再生")
        ctrl_layout = QHBoxLayout(ctrl_group)
        self.btn_play = QPushButton("再生")
        self.btn_pause = QPushButton("一時停止")
        self.btn_stop = QPushButton("停止")
        self.btn_play.clicked.connect(self.start_play)
        self.btn_pause.clicked.connect(self.pause_play)
        self.btn_stop.clicked.connect(self.stop_play)
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addStretch()
        scroll_layout.addWidget(ctrl_group)

        # シークスライダー
        seek_layout = QHBoxLayout()
        self.label_time = QLabel("0:00 / 0:00")
        self.slider_seek = QSlider(Qt.Horizontal)
        self.slider_seek.setMinimum(0)
        self.slider_seek.setMaximum(0)
        self.slider_seek.sliderMoved.connect(self.on_seek)
        seek_layout.addWidget(self.label_time)
        seek_layout.addWidget(self.slider_seek, 1)
        scroll_layout.addLayout(seek_layout)

        # コマ送り（左: 前へ / 右: 次へ）
        step_group = QGroupBox("コマ送り")
        step_layout = QHBoxLayout(step_group)
        self.step_buttons = []
        for label, delta in FRAME_STEP_OPTIONS:
            btn = QPushButton(label)
            btn.setFixedWidth(48)
            btn.setStyleSheet("font-weight: bold; font-size: 13px;")
            btn.clicked.connect(lambda checked=False, d=delta: self.do_frame_step(d))
            step_layout.addWidget(btn)
            self.step_buttons.append(btn)
        step_layout.addStretch()
        scroll_layout.addWidget(step_group)

        # キャプチャ
        cap_group = QGroupBox("キャプチャ")
        cap_layout = QHBoxLayout(cap_group)
        self.btn_capture = QPushButton("現在のフレームを画像として保存")
        self.btn_capture.clicked.connect(self.capture_frame)
        cap_layout.addWidget(self.btn_capture)
        scroll_layout.addWidget(cap_group)

        # トリミング（一時停止時のみ有効。ドラッグ＋数値入力で範囲指定）
        trim_group = QGroupBox("トリミング（一時停止時のみ）")
        trim_layout = QVBoxLayout(trim_group)
        trim_layout.addWidget(QLabel("範囲をドラッグまたは数値で指定:"))
        # 開始・終了をドラッグで指定するスライダー
        trim_slider_layout = QHBoxLayout()
        trim_slider_layout.addWidget(QLabel("開始", minimumWidth=32))
        self.slider_trim_in = QSlider(Qt.Horizontal)
        self.slider_trim_in.setMinimum(0)
        self.slider_trim_in.setMaximum(0)
        self.slider_trim_in.valueChanged.connect(self._on_trim_in_slider)
        trim_slider_layout.addWidget(self.slider_trim_in, 1)
        trim_layout.addLayout(trim_slider_layout)
        trim_slider_layout2 = QHBoxLayout()
        trim_slider_layout2.addWidget(QLabel("終了", minimumWidth=32))
        self.slider_trim_out = QSlider(Qt.Horizontal)
        self.slider_trim_out.setMinimum(0)
        self.slider_trim_out.setMaximum(0)
        self.slider_trim_out.valueChanged.connect(self._on_trim_out_slider)
        trim_slider_layout2.addWidget(self.slider_trim_out, 1)
        trim_layout.addLayout(trim_slider_layout2)
        # 数値入力（スライダーと併用）
        trim_input_layout = QHBoxLayout()
        trim_input_layout.addWidget(QLabel("開始:"))
        self.spin_in = QSpinBox()
        self.spin_in.setMinimum(0)
        self.spin_in.setMaximum(0)
        self.spin_in.valueChanged.connect(self._on_trim_in_spin)
        trim_input_layout.addWidget(self.spin_in)
        trim_input_layout.addWidget(QLabel("終了:"))
        self.spin_out = QSpinBox()
        self.spin_out.setMinimum(0)
        self.spin_out.setMaximum(0)
        self.spin_out.valueChanged.connect(self._on_trim_out_spin)
        trim_input_layout.addWidget(self.spin_out)
        self.btn_set_in = QPushButton("ここを開始に")
        self.btn_set_out = QPushButton("ここを終了に")
        self.btn_set_in.clicked.connect(self.set_in_point)
        self.btn_set_out.clicked.connect(self.set_out_point)
        trim_input_layout.addWidget(self.btn_set_in)
        trim_input_layout.addWidget(self.btn_set_out)
        trim_input_layout.addStretch()
        trim_layout.addLayout(trim_input_layout)
        self.btn_export_trim = QPushButton("トリミングして保存")
        self.btn_export_trim.clicked.connect(self.export_trimmed)
        trim_layout.addWidget(self.btn_export_trim)
        scroll_layout.addWidget(trim_group)
        self.trim_group = trim_group

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        self.update_ui_state(False)

    def update_ui_state(self, has_video: bool):
        self.btn_play.setEnabled(has_video)
        self.btn_pause.setEnabled(has_video)
        self.btn_stop.setEnabled(has_video)
        self.slider_seek.setEnabled(has_video)
        for btn in self.step_buttons:
            btn.setEnabled(has_video)
        self.btn_capture.setEnabled(has_video)
        self._update_trim_ui_state()

    def _update_trim_ui_state(self):
        """トリミングは動画読み込み済みかつ一時停止時のみ有効"""
        can_trim = self.cap is not None and not self.playing
        self.trim_group.setEnabled(can_trim)
        if can_trim:
            self.trim_group.setTitle("トリミング（一時停止中: 範囲をドラッグまたは数値で指定）")
        else:
            self.trim_group.setTitle("トリミング（一時停止時のみ利用できます）")

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "動画を開く", self._last_video_dir,
            "Video (*.mp4 *.avi *.mov *.mkv *.wmv *.webm);;All (*.*)"
        )
        if not path:
            return
        self._last_video_dir = str(Path(path).parent)
        QSettings("template_matching", "video_tool").setValue("lastVideoDir", self._last_video_dir)
        self._close_capture()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "エラー", "動画を開けませんでした。")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_path = path
        self.current_frame_index = 0
        self.in_point = 0
        self.out_point = max(0, self.total_frames - 1)
        self.label_path.setText(Path(path).name)
        self.slider_seek.setMaximum(max(0, self.total_frames - 1))
        self.slider_trim_in.setMaximum(max(0, self.total_frames - 1))
        self.slider_trim_out.setMaximum(max(0, self.total_frames - 1))
        self.spin_in.setMaximum(self.total_frames - 1)
        self.spin_out.setMaximum(self.total_frames - 1)
        self.spin_in.setValue(0)
        self.spin_out.setValue(self.out_point)
        self.slider_trim_in.setValue(0)
        self.slider_trim_out.setValue(self.out_point)
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
        self.update_ui_state(False)
        self.label_video.setText("動画を開いてください")
        self.label_video.setStyleSheet("background: #1a1a1a; color: #666; font-size: 14px;")

    def _frame_to_time(self, frame_index: int) -> str:
        if self.fps <= 0:
            return "0:00"
        sec = frame_index / self.fps
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}:{s:02d}"

    def _update_time_label(self):
        if self.cap is None:
            self.label_time.setText("0:00 / 0:00")
            return
        total_time = self._frame_to_time(self.total_frames)
        current_time = self._frame_to_time(self.current_frame_index)
        self.label_time.setText(f"{current_time} / {total_time}")
        self.slider_seek.blockSignals(True)
        self.slider_seek.setValue(self.current_frame_index)
        self.slider_seek.blockSignals(False)

    def _display_size(self):
        w = self.label_video.width()
        h = self.label_video.height()
        if w <= 0 or h <= 0:
            return 0, 0
        return w, h

    def _show_frame_at(self, frame_index: int):
        if self.cap is None or self.total_frames == 0:
            return
        frame_index = max(0, min(frame_index, self.total_frames - 1))
        self.current_frame_index = frame_index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret and frame is not None:
            sw, sh = self._display_size()
            pix = cv2_to_qpixmap(frame, sw, sh, fast=False)
            self.label_video.setPixmap(pix)
            self.label_video.setStyleSheet("")
        self._update_time_label()

    def on_seek(self, value: int):
        self._show_frame_at(value)

    def _stop_playback_thread(self):
        if self.playback_thread and self.playback_thread.isRunning():
            self.playback_thread.stop()
            self.playback_thread.wait(2000)
        self.playback_thread = None

    def _on_playback_frame(self, frame_index: int, frame: np.ndarray):
        self.current_frame_index = frame_index
        sw, sh = self._display_size()
        pix = cv2_to_qpixmap(frame, sw, sh, fast=True)
        self.label_video.setPixmap(pix)
        self.label_video.setStyleSheet("")
        self._update_time_label()

    def _on_playback_finished(self):
        self.playing = False
        self.playback_thread = None
        self._update_time_label()
        self._update_trim_ui_state()

    def start_play(self):
        if self.cap is None or not self.video_path or self.total_frames == 0:
            return
        self._stop_playback_thread()
        self.playing = True
        self._update_trim_ui_state()
        self.playback_thread = PlaybackThread(
            self.video_path, self.fps, self.total_frames
        )
        self.playback_thread.set_start_frame(self.current_frame_index)
        self.playback_thread.frame_ready.connect(self._on_playback_frame)
        self.playback_thread.finished_playback.connect(self._on_playback_finished)
        self.playback_thread.start()

    def pause_play(self):
        self.playing = False
        self._stop_playback_thread()
        self._update_trim_ui_state()

    def stop_play(self):
        self.pause_play()
        self._show_frame_at(0)

    def do_frame_step(self, delta: int):
        if self.cap is None:
            return
        next_index = self.current_frame_index + delta
        next_index = max(0, min(next_index, self.total_frames - 1))
        self._show_frame_at(next_index)

    def _sync_trim_in_out(self):
        """開始 > 終了にならないようそろえる（スライダーとスピンは連動済み想定）"""
        a, b = self.spin_in.value(), self.spin_out.value()
        if a > b:
            self.spin_out.blockSignals(True)
            self.slider_trim_out.blockSignals(True)
            self.spin_out.setValue(a)
            self.slider_trim_out.setValue(a)
            self.spin_out.blockSignals(False)
            self.slider_trim_out.blockSignals(False)
        elif b < a:
            self.spin_in.blockSignals(True)
            self.slider_trim_in.blockSignals(True)
            self.spin_in.setValue(b)
            self.slider_trim_in.setValue(b)
            self.spin_in.blockSignals(False)
            self.slider_trim_in.blockSignals(False)
        self.in_point = self.spin_in.value()
        self.out_point = self.spin_out.value()

    def _on_trim_in_slider(self, value: int):
        self.spin_in.blockSignals(True)
        self.spin_in.setValue(value)
        self.spin_in.blockSignals(False)
        self._sync_trim_in_out()

    def _on_trim_out_slider(self, value: int):
        self.spin_out.blockSignals(True)
        self.spin_out.setValue(value)
        self.spin_out.blockSignals(False)
        self._sync_trim_in_out()

    def _on_trim_in_spin(self, value: int):
        self.slider_trim_in.blockSignals(True)
        self.slider_trim_in.setValue(value)
        self.slider_trim_in.blockSignals(False)
        self._sync_trim_in_out()

    def _on_trim_out_spin(self, value: int):
        self.slider_trim_out.blockSignals(True)
        self.slider_trim_out.setValue(value)
        self.slider_trim_out.blockSignals(False)
        self._sync_trim_in_out()

    def set_in_point(self):
        v = self.current_frame_index
        self.spin_in.setValue(v)
        self.slider_trim_in.setValue(v)
        self.in_point = v
        self._sync_trim_in_out()

    def set_out_point(self):
        v = self.current_frame_index
        self.spin_out.setValue(v)
        self.slider_trim_out.setValue(v)
        self.out_point = v
        self._sync_trim_in_out()

    def capture_frame(self):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            QMessageBox.warning(self, "キャプチャ", "フレームを取得できませんでした。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "画像を保存", "",
            "PNG (*.png);;JPEG (*.jpg);;All (*.*)"
        )
        if not path:
            return
        if cv2.imwrite(path, frame):
            QMessageBox.information(self, "キャプチャ", f"保存しました: {path}")
        else:
            QMessageBox.critical(self, "キャプチャ", "保存に失敗しました。")

    def export_trimmed(self):
        if self.cap is None:
            return
        self._sync_trim_in_out()
        start_f = self.spin_in.value()
        end_f = self.spin_out.value()
        if start_f > end_f:
            QMessageBox.warning(self, "トリミング", "開始フレームが終了より後です。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "トリミング動画を保存", "",
            "MP4 (*.mp4);;AVI (*.avi);;All (*.*)"
        )
        if not path:
            return
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if path.lower().endswith(".avi"):
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if not out.isOpened():
            QMessageBox.critical(self, "トリミング", "動画の書き出しを開始できませんでした。")
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for _ in range(end_f - start_f + 1):
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            out.write(frame)
        out.release()
        QMessageBox.information(self, "トリミング", f"保存しました: {path}")

    def closeEvent(self, event):
        self._close_capture()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = VideoToolWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
