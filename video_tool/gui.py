"""
動画ツール: 取り込み・再生/一時停止・コマ送り・キャプチャ・トリミング
"""
import sys
import time
from pathlib import Path

import numpy as np  # OpenCV より先に import
import cv2
from PySide6.QtCore import Qt, QThread, Signal, QSettings, QEvent, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStyleFactory,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QFrame,
    QLayout,
    QSplitter,
    QPushButton,
    QLabel,
    QSlider,
    QFileDialog,
    QGroupBox,
    QSpinBox,
    QComboBox,
    QMessageBox,
)

class ClickableSlider(QSlider):
    """トラックをクリックでジャンプ、つまみをドラッグで移動できるスライダー（スタイルに依存しない）"""

    def _value_to_x(self, value):
        """値に対応するつまみ中心の x 位置（ウィジェット座標）"""
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
        handle_half = 10
        # つまみの上ならドラッグに任せる
        if abs(pos_x - handle_center) <= handle_half:
            super().mousePressEvent(event)
            return
        # トラックをクリック → その位置へジャンプ
        w = self.width()
        margin = 12
        span = max(1, w - 2 * margin)
        val = self.minimum() + (self.maximum() - self.minimum()) * (pos_x - margin) / span
        self.setValue(int(max(self.minimum(), min(self.maximum(), val))))
        event.accept()


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
        self.crop_left = 0
        self.crop_top = 0
        self.crop_width = 0
        self.crop_height = 0
        self._frame_width = 0
        self._frame_height = 0
        self._current_frame_np = None
        self._dragging_crop = False
        self._drag_start = None
        self._drag_current = None
        self._disp_pix_w = 0
        self._disp_pix_h = 0
        self._disp_off_x = 0
        self._disp_off_y = 0
        self._pending_playback_frame = None
        self._pending_playback_index = -1
        self._last_paint_time = 0.0
        _settings = QSettings("template_matching", "video_tool")
        self._last_video_dir = _settings.value("lastVideoDir", "", type=str)
        self._last_capture_dir = _settings.value("lastCaptureDir", "", type=str)
        _templates = Path(__file__).resolve().parent.parent / "templates"
        self._templates_dir = str(_templates)
        self._template_categories = ("bonus", "fever", "go", "result", "skill", "timeup")
        self.setWindowTitle("動画ツール")
        self.setMinimumSize(980, 1000)
        self.resize(1020, 1040)
        self.setup_ui()


    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setSizeConstraint(QLayout.SetMinimumSize)
        main.setSpacing(8)
        main.setContentsMargins(8, 8, 8, 8)

        # 左右2分割
        splitter = QSplitter(Qt.Horizontal)

        # === 左エリア: 動画を開く → 動画 → スライダー → 再生等 → コマ送り ===
        left = QWidget()
        left.setMinimumWidth(560)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        open_area = QFrame()
        open_area.setFrameStyle(QFrame.StyledPanel)
        open_area.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        open_area.setMinimumHeight(48)
        open_area_layout = QHBoxLayout(open_area)
        open_area_layout.setContentsMargins(6, 6, 6, 6)
        self.btn_open = QPushButton("動画を開く")
        self.label_path = QLabel("（未選択）")
        self.label_path.setStyleSheet("color: #666;")
        open_area_layout.addWidget(self.btn_open)
        open_area_layout.addWidget(self.label_path, 1)
        left_layout.addWidget(open_area)
        self.btn_open.clicked.connect(self.open_video)

        video_area = QFrame()
        video_area.setFrameStyle(QFrame.StyledPanel)
        video_area.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        video_area.setMinimumSize(560, 400)
        video_area.setMinimumHeight(400)
        video_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl = QVBoxLayout(video_area)
        vl.setContentsMargins(0, 0, 0, 0)
        self.label_video = QLabel()
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setStyleSheet("color: #999;")
        self.label_video.setText("動画を開いてください")
        self.label_video.setMouseTracking(True)
        self.label_video.installEventFilter(self)
        self.label_video.setScaledContents(False)
        self.label_video.setMaximumSize(1920, 1080)
        self.label_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
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
        self.btn_play.clicked.connect(self.start_play)
        self.btn_pause.clicked.connect(self.pause_play)
        self.btn_stop.clicked.connect(self.stop_play)
        ctrl_area_layout.addWidget(self.btn_play)
        ctrl_area_layout.addWidget(self.btn_pause)
        ctrl_area_layout.addWidget(self.btn_stop)
        ctrl_area_layout.addWidget(QLabel("速度:"))
        ctrl_area_layout.addWidget(self.combo_speed)
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
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

        # === 右エリア: トリミング ===
        right = QWidget()
        right.setMinimumWidth(260)
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_open_area = QFrame()
        right_open_area.setFrameStyle(QFrame.StyledPanel)
        right_open_area.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        right_open_area.setMinimumHeight(48)
        right_open_area_layout = QHBoxLayout(right_open_area)
        right_open_area_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.addWidget(right_open_area)
        right_video_area = QFrame()
        right_video_area.setFrameStyle(QFrame.StyledPanel)
        right_video_area.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        right_video_area.setMinimumSize(260, 400)
        right_video_area.setMinimumHeight(400)
        right_video_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_video_layout = QVBoxLayout(right_video_area)
        right_video_layout.setContentsMargins(0, 0, 0, 0)
        self.label_crop_preview = QLabel()
        self.label_crop_preview.setMinimumSize(384, 216)
        self.label_crop_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_crop_preview.setAlignment(Qt.AlignCenter)
        self.label_crop_preview.setStyleSheet("color: #999;")
        self.label_crop_preview.setText("（範囲指定で表示）")
        self.label_crop_preview.setScaledContents(False)
        right_video_layout.setContentsMargins(12, 12, 12, 12)
        right_video_layout.addWidget(self.label_crop_preview, 1)
        right_layout.addWidget(right_video_area)
        trim_group = QGroupBox("トリミング")
        trim_group.setStyleSheet("QGroupBox { border: 1px solid #ddd; margin-top: 8px; padding-top: 8px; }")
        self.trim_group = trim_group
        trim_layout = QVBoxLayout(trim_group)
        trim_layout.setSpacing(10)
        tr1 = QHBoxLayout()
        tr1.setSpacing(6)
        tr1.addStretch()
        tr1.addWidget(QLabel("左:"))
        self.spin_crop_left = QSpinBox()
        self.spin_crop_left.setMinimum(0)
        self.spin_crop_left.setMaximum(0)
        self.spin_crop_left.valueChanged.connect(self._on_crop_spin_changed)
        tr1.addWidget(self.spin_crop_left)
        tr1.addWidget(QLabel("上:"))
        self.spin_crop_top = QSpinBox()
        self.spin_crop_top.setMinimum(0)
        self.spin_crop_top.setMaximum(0)
        self.spin_crop_top.valueChanged.connect(self._on_crop_spin_changed)
        tr1.addWidget(self.spin_crop_top)
        tr1.addWidget(QLabel("幅:"))
        self.spin_crop_width = QSpinBox()
        self.spin_crop_width.setMinimum(1)
        self.spin_crop_width.setMaximum(0)
        self.spin_crop_width.valueChanged.connect(self._on_crop_spin_changed)
        tr1.addWidget(self.spin_crop_width)
        tr1.addWidget(QLabel("高さ:"))
        self.spin_crop_height = QSpinBox()
        self.spin_crop_height.setMinimum(1)
        self.spin_crop_height.setMaximum(0)
        self.spin_crop_height.valueChanged.connect(self._on_crop_spin_changed)
        tr1.addWidget(self.spin_crop_height)
        tr1.addStretch()
        trim_layout.addLayout(tr1)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_export_trim_image = QPushButton("トリミング範囲を画像で保存")
        self.btn_export_trim_image.clicked.connect(self.export_trim_as_image)
        btn_row.addWidget(self.btn_export_trim_image)
        btn_row.addStretch()
        trim_layout.addLayout(btn_row)
        right_layout.addWidget(trim_group)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 500])

        # === 下: 現在のフレームを保存 ===
        bottom_bar = QFrame()
        bottom_bar.setFrameStyle(QFrame.StyledPanel)
        bottom_bar.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(8, 8, 8, 8)
        self.btn_capture = QPushButton("現在のフレームを保存")
        self.btn_capture.clicked.connect(self.capture_frame)
        bottom_layout.addWidget(self.btn_capture)
        bottom_layout.addStretch()

        # 縦スプリッター: 上(1+2) 9 : 下(5) 1
        vert_splitter = QSplitter(Qt.Vertical)
        vert_splitter.addWidget(splitter)
        vert_splitter.addWidget(bottom_bar)
        vert_splitter.setStretchFactor(0, 9)
        vert_splitter.setStretchFactor(1, 1)
        vert_splitter.setSizes([900, 100])
        main.addWidget(vert_splitter)

        self.update_ui_state(False)

    def update_ui_state(self, has_video: bool):
        self.btn_play.setEnabled(has_video)
        self.btn_pause.setEnabled(has_video)
        self.btn_stop.setEnabled(has_video)
        self.combo_speed.setEnabled(has_video)
        self.slider_seek.setEnabled(has_video)
        for btn in self.step_buttons:
            btn.setEnabled(has_video)
        self.btn_capture.setEnabled(has_video)
        self._update_trim_ui_state()

    def _update_trim_ui_state(self):
        """トリミングは動画読み込み済みかつ一時停止時のみ有効"""
        can_trim = self.cap is not None and not self.playing
        self.trim_group.setEnabled(can_trim)
        self.trim_group.setTitle("トリミング（一時停止時のみ）" if not can_trim else "トリミング")

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
        self._frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.crop_left = 0
        self.crop_top = 0
        self.crop_width = max(1, self._frame_width)
        self.crop_height = max(1, self._frame_height)
        p = Path(path)
        self.label_path.setText(p.name)
        self.label_path.setToolTip(str(p.resolve()))
        self.slider_seek.setMaximum(max(0, self.total_frames - 1))
        self.spin_crop_left.setMaximum(max(0, self._frame_width - 1))
        self.spin_crop_top.setMaximum(max(0, self._frame_height - 1))
        self.spin_crop_width.setMaximum(self._frame_width)
        self.spin_crop_height.setMaximum(self._frame_height)
        self.spin_crop_left.setValue(0)
        self.spin_crop_top.setValue(0)
        self.spin_crop_width.setValue(self.crop_width)
        self.spin_crop_height.setValue(self.crop_height)
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
        self.label_video.setStyleSheet("color: #666; font-size: 14px;")
        self.label_crop_preview.clear()
        self.label_crop_preview.setText("（範囲指定で表示）")
        self.label_crop_preview.setStyleSheet("color: #666;")

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
        """動画表示用サイズ。枠が再生で大きくなり続けないよう上限をかける"""
        w = self.label_video.width()
        h = self.label_video.height()
        if w <= 0 or h <= 0:
            return 0, 0
        w = min(w, 1920)
        h = min(h, 1080)
        return w, h

    def _label_to_video_coords(self, lx: float, ly: float):
        """ラベル上の座標を動画フレーム座標に変換"""
        if self._disp_pix_w <= 0 or self._frame_width <= 0:
            return 0, 0
        px = lx - self._disp_off_x
        py = ly - self._disp_off_y
        vx = int(px * self._frame_width / self._disp_pix_w)
        vy = int(py * self._frame_height / self._disp_pix_h)
        return max(0, min(self._frame_width, vx)), max(0, min(self._frame_height, vy))

    def _show_frame_at(self, frame_index: int):
        if self.cap is None or self.total_frames == 0:
            return
        frame_index = max(0, min(frame_index, self.total_frames - 1))
        self.current_frame_index = frame_index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self._current_frame_np = frame.copy()
            self._paint_frame_with_crop()
        self._update_time_label()
        self._update_crop_preview()

    def on_seek(self, value: int):
        self._show_frame_at(value)

    def _paint_frame_with_crop(self, fast: bool = False):
        """現在フレームに crop 矩形（またはドラッグ中なら仮矩形）を描画して表示"""
        if self._current_frame_np is None:
            return
        img = self._current_frame_np.copy()
        if self._dragging_crop and self._drag_start is not None and self._drag_current is not None:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_current
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
            l, t = self.crop_left, self.crop_top
            r, b = l + self.crop_width, t + self.crop_height
            cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
        sw, sh = self._display_size()
        pix = cv2_to_qpixmap(img, sw, sh, fast=fast)
        self._disp_pix_w = pix.width()
        self._disp_pix_h = pix.height()
        self._disp_off_x = (self.label_video.width() - pix.width()) / 2
        self._disp_off_y = (self.label_video.height() - pix.height()) / 2
        self.label_video.setPixmap(pix)
        self.label_video.setStyleSheet("")

    def _update_crop_preview(self):
        """トリミング範囲を切り出してプレビューに表示"""
        if self._current_frame_np is None or self.crop_width <= 0 or self.crop_height <= 0:
            return
        h, w = self._current_frame_np.shape[:2]
        l = max(0, min(self.crop_left, w - 1))
        t = max(0, min(self.crop_top, h - 1))
        r = min(w, l + self.crop_width)
        b = min(h, t + self.crop_height)
        if r <= l or b <= t:
            return
        cropped = self._current_frame_np[t:b, l:r]
        pw = max(256, self.label_crop_preview.width())
        ph = max(144, self.label_crop_preview.height())
        pix = cv2_to_qpixmap(cropped, pw, ph, fast=True)
        self.label_crop_preview.setPixmap(pix)
        self.label_crop_preview.setStyleSheet("")

    def _on_crop_spin_changed(self):
        if self._frame_width <= 0 or self._frame_height <= 0:
            return
        self.crop_left = self.spin_crop_left.value()
        self.crop_top = self.spin_crop_top.value()
        self.crop_width = max(1, self.spin_crop_width.value())
        self.crop_height = max(1, self.spin_crop_height.value())
        self.crop_left = max(0, min(self.crop_left, self._frame_width - 1))
        self.crop_top = max(0, min(self.crop_top, self._frame_height - 1))
        self.crop_width = min(self.crop_width, self._frame_width - self.crop_left)
        self.crop_height = min(self.crop_height, self._frame_height - self.crop_top)
        self.spin_crop_left.blockSignals(True)
        self.spin_crop_top.blockSignals(True)
        self.spin_crop_width.blockSignals(True)
        self.spin_crop_height.blockSignals(True)
        self.spin_crop_left.setValue(self.crop_left)
        self.spin_crop_top.setValue(self.crop_top)
        self.spin_crop_width.setValue(self.crop_width)
        self.spin_crop_height.setValue(self.crop_height)
        self.spin_crop_left.blockSignals(False)
        self.spin_crop_top.blockSignals(False)
        self.spin_crop_width.blockSignals(False)
        self.spin_crop_height.blockSignals(False)
        if self._current_frame_np is not None:
            self._paint_frame_with_crop()
            self._update_crop_preview()

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
        self._paint_frame_with_crop(fast=True)
        self._update_time_label()
        if not self.playing:
            self._update_crop_preview()

    def _on_playback_finished(self):
        self.playing = False
        self.playback_thread = None
        self._flush_playback_frame()
        self._update_time_label()
        self._update_trim_ui_state()

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
        self._update_trim_ui_state()
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
        self._update_trim_ui_state()

    def eventFilter(self, obj, event):
        if getattr(self, "slider_seek", None) is not None and obj == self.slider_seek and event.type() == QEvent.MouseButtonPress and self.playing:
            self.pause_play()
        if obj == self.label_video and getattr(self, "cap", None) is not None and not self.playing:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._dragging_crop = True
                self._drag_start = self._label_to_video_coords(event.position().x(), event.position().y())
                self._drag_current = self._drag_start
                return True
            if event.type() == QEvent.MouseMove:
                if self._dragging_crop and self._drag_start is not None:
                    self._drag_current = self._label_to_video_coords(event.position().x(), event.position().y())
                    self._paint_frame_with_crop()
                return False
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self._dragging_crop:
                self._drag_current = self._label_to_video_coords(event.position().x(), event.position().y())
                x1, y1 = self._drag_start
                x2, y2 = self._drag_current
                left = max(0, min(x1, x2))
                top = max(0, min(y1, y2))
                right = min(self._frame_width, max(x1, x2))
                bottom = min(self._frame_height, max(y1, y2))
                w = max(1, right - left)
                h = max(1, bottom - top)
                self._dragging_crop = False
                self._drag_start = None
                self._drag_current = None
                self.spin_crop_left.blockSignals(True)
                self.spin_crop_top.blockSignals(True)
                self.spin_crop_width.blockSignals(True)
                self.spin_crop_height.blockSignals(True)
                self.spin_crop_left.setValue(left)
                self.spin_crop_top.setValue(top)
                self.spin_crop_width.setValue(w)
                self.spin_crop_height.setValue(h)
                self.spin_crop_left.blockSignals(False)
                self.spin_crop_top.blockSignals(False)
                self.spin_crop_width.blockSignals(False)
                self.spin_crop_height.blockSignals(False)
                self.crop_left = left
                self.crop_top = top
                self.crop_width = w
                self.crop_height = h
                self._paint_frame_with_crop()
                self._update_crop_preview()
                return True
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

    def _next_save_path(self, category: str, subdir: str, ext: str) -> str:
        """category/frames または category/trimmed で既存の XXX_001 を調べ、次の番号のパスを返す"""
        dir_path = Path(self._templates_dir) / category / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        prefix = f"{category}_"
        max_n = 0
        for f in dir_path.iterdir():
            if not f.is_file():
                continue
            if f.name.startswith(prefix) and f.suffix:
                try:
                    n = int(f.stem[len(prefix):])
                    if n > max_n:
                        max_n = n
                except ValueError:
                    pass
        return str(dir_path / f"{prefix}{max_n + 1:03d}{ext}")

    def _next_capture_path(self, dir_path: Path, prefix: str, ext: str) -> str:
        """指定フォルダ内で prefix_001, prefix_002 の次の番号のパスを返す"""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        head = f"{prefix}_"
        max_n = 0
        for f in dir_path.iterdir():
            if not f.is_file():
                continue
            if f.stem.startswith(head) and f.suffix:
                try:
                    n = int(f.stem[len(head):])
                    if n > max_n:
                        max_n = n
                except ValueError:
                    pass
        return str(dir_path / f"{head}{max_n + 1:03d}{ext}")

    def capture_frame(self):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            QMessageBox.warning(self, "キャプチャ", "フレームを取得できませんでした。")
            return
        if self._last_capture_dir and Path(self._last_capture_dir).is_dir():
            default_path = self._next_capture_path(Path(self._last_capture_dir), "frame", ".png")
        else:
            default_path = self._next_save_path("bonus", "frames", ".png")
        path, _ = QFileDialog.getSaveFileName(
            self, "画像を保存", default_path,
            "PNG (*.png);;JPEG (*.jpg);;All (*.*)"
        )
        if not path:
            return
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = (p.suffix.lower() or ".png").replace(".jpeg", ".jpg")
        if ext not in (".png", ".jpg"):
            ext = ".png"
        _, buf = cv2.imencode(ext, frame)
        try:
            if buf is not None:
                p.write_bytes(buf.tobytes())
                self._last_capture_dir = str(p.parent)
                QSettings("template_matching", "video_tool").setValue("lastCaptureDir", self._last_capture_dir)
                QMessageBox.information(self, "キャプチャ", f"保存しました: {p}")
            else:
                QMessageBox.critical(self, "キャプチャ", "保存に失敗しました。")
        except OSError as e:
            QMessageBox.critical(self, "キャプチャ", f"保存に失敗しました: {e}")

    def export_trim_as_image(self):
        """トリミング範囲を画像（PNG）で保存"""
        if self.cap is None or self._current_frame_np is None:
            QMessageBox.warning(self, "トリミング", "動画を開き、フレームを表示してください。")
            return
        if self.crop_width <= 0 or self.crop_height <= 0:
            QMessageBox.warning(self, "トリミング", "範囲を指定してください。")
            return
        h, w = self._current_frame_np.shape[:2]
        l = max(0, min(self.crop_left, w - 1))
        t = max(0, min(self.crop_top, h - 1))
        r = min(w, l + self.crop_width)
        b = min(h, t + self.crop_height)
        if r <= l or b <= t:
            return
        cropped = self._current_frame_np[t:b, l:r]
        if self._last_capture_dir and Path(self._last_capture_dir).is_dir():
            default_path = self._next_capture_path(Path(self._last_capture_dir), "trim", ".png")
        else:
            default_path = self._next_save_path("bonus", "frames", ".png")
        path, _ = QFileDialog.getSaveFileName(
            self, "トリミング範囲を画像で保存", default_path,
            "PNG (*.png);;JPEG (*.jpg);;All (*.*)"
        )
        if not path:
            return
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = (p.suffix.lower() or ".png").replace(".jpeg", ".jpg")
        if ext not in (".png", ".jpg"):
            ext = ".png"
        _, buf = cv2.imencode(ext, cropped)
        try:
            if buf is not None:
                p.write_bytes(buf.tobytes())
                self._last_capture_dir = str(p.parent)
                QSettings("template_matching", "video_tool").setValue("lastCaptureDir", self._last_capture_dir)
                QMessageBox.information(self, "トリミング", f"保存しました: {p}")
            else:
                QMessageBox.critical(self, "トリミング", "保存に失敗しました。")
        except OSError as e:
            QMessageBox.critical(self, "トリミング", f"保存に失敗しました: {e}")

    def closeEvent(self, event):
        self._close_capture()
        event.accept()


def _get_platform_stylesheet():
    """Mac のときは Fusion を Mac 風に見せるスタイル。Windows では空."""
    if sys.platform != "darwin":
        return ""
    return """
        QMainWindow, QWidget { background-color: #f5f5f7; }
        QPushButton { background-color: #fff; border: 1px solid #d2d2d7; border-radius: 6px; padding: 6px 12px; }
        QPushButton:hover { background-color: #e8e8ed; }
        QPushButton:pressed { background-color: #d2d2d7; }
        QGroupBox { font-weight: 600; color: #1d1d1f; border: 1px solid #d2d2d7; border-radius: 6px; margin-top: 10px; padding: 8px 8px 4px 8px; background: #f5f5f7; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QFrame { border: 1px solid #d2d2d7; border-radius: 4px; background-color: #fff; }
        QSlider::groove:horizontal { height: 6px; border-radius: 3px; background: #e8e8ed; }
        QSlider::handle:horizontal { width: 16px; margin: -5px 0; border-radius: 8px; background: #fff; border: 1px solid #d2d2d7; }
        QSlider::handle:horizontal:hover { background: #e8e8ed; }
        QSpinBox, QComboBox { background: #fff; border: 1px solid #d2d2d7; border-radius: 6px; padding: 4px 8px; min-height: 20px; }
        QLabel { color: #1d1d1f; }
    """


def main():
    app = QApplication(sys.argv)
    if sys.platform == "darwin":
        mac_style = QStyleFactory.create("macos")
        if mac_style is not None:
            app.setStyle(mac_style)
        else:
            app.setStyle("Fusion")
            app.setStyleSheet(_get_platform_stylesheet())
    else:
        app.setStyle("Fusion")
    w = VideoToolWindow()
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
