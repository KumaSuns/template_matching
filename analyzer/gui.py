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
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
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

# テンプレート画像: templates からの相対パス
TEMPLATE_SOURCE_PATHS = ["go/frames", "timeup/frames", "result/frames"]


def _get_templates_base_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "templates"


def load_templates(
    threshold_go: float | None = None,
    threshold_timeup: float | None = None,
    threshold_result: float | None = None,
) -> list[tuple[str, np.ndarray, float]]:
    """(表示名, BGR, 閾値) のリストで返す。連続ヒットは1件にまとめる。"""
    base = _get_templates_base_dir()
    settings = QSettings("analyzer", "main")
    th_go = threshold_go if threshold_go is not None else float(settings.value("detectThreshold", 0.75, type=float))
    th_timeup = threshold_timeup if threshold_timeup is not None else float(settings.value("detectThresholdTimeup", 0.75, type=float))
    th_result = threshold_result if threshold_result is not None else float(settings.value("detectThresholdResult", 0.75, type=float))
    config = {
        "go/frames": th_go,
        "timeup/frames": th_timeup,
        "result/frames": th_result,
    }
    result = []
    for path_suffix in TEMPLATE_SOURCE_PATHS:
        threshold = config.get(path_suffix, 0.75)
        templates_dir = base / path_suffix
        if not templates_dir.is_dir():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for path in sorted(templates_dir.glob(ext)):
                img = cv2.imread(str(path))
                if img is not None and img.size > 0:
                    display_name = f"{path_suffix}/{path.name}"
                    result.append((display_name, img, threshold))
    return result


def match_template_score(frame: np.ndarray, template: np.ndarray) -> float:
    """
    フレームとテンプレートの類似度を 0～1 で返す（cv2.matchTemplate TM_CCOEFF_NORMED）。
    フレームがテンプレートより小さい場合はフレームをリサイズしてから比較する。
    """
    hf, wf = frame.shape[:2]
    ht, wt = template.shape[:2]
    if wf < wt or hf < ht:
        scale = min(wt / max(1, wf), ht / max(1, hf))
        new_w = max(wt, int(wf * scale))
        new_h = max(ht, int(hf * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        hf, wf = frame.shape[:2]
    if len(frame.shape) == 3 and len(template.shape) == 3:
        pass
    elif len(frame.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    elif len(template.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    try:
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max(0.0, min(1.0, max_val)))
    except cv2.error:
        return 0.0

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


class DetectThread(QThread):
    """動画を走査し、テンプレートに近いフレームを検出する"""
    progress = Signal(int, int)  # current, total
    result_item = Signal(int, str, float)  # frame_index, template_name, score
    finished_detect = Signal()

    def __init__(self, video_path: str, total_frames: int, fps: float, templates: list[tuple[str, np.ndarray, float]], frame_step: int, use_local_peak: bool = False):
        super().__init__()
        self.video_path = video_path
        self.total_frames = total_frames
        self.fps = max(1.0, fps)
        self.templates = templates  # (name, image, threshold)
        self.frame_step = max(1, frame_step)
        self.use_local_peak = use_local_peak
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        self._stop = False
        if not self.templates:
            self.finished_detect.emit()
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_detect.emit()
            return
        try:
            num_steps = (self.total_frames + self.frame_step - 1) // self.frame_step
            step_index = 0
            prev_matched: dict[str, bool] = {}
            sources = {name.rsplit("/", 1)[0] for name, *_ in self.templates}
            # ローカル最大のみ: ソースごとに (frame_index, score, name) を最大3件保持し、中央がピークのときだけ報告
            peak_buf: dict[str, list[tuple[int, float, str]]] = {src: [] for src in sources} if self.use_local_peak else {}

            for frame_index in range(0, self.total_frames, self.frame_step):
                if self._stop:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret or frame is None:
                    step_index += 1
                    self.progress.emit(step_index, num_steps)
                    continue
                best_per_source: dict[str, tuple[float, str]] = {}
                for name, template, threshold in self.templates:
                    if self._stop:
                        break
                    score = match_template_score(frame, template)
                    if score >= threshold:
                        source = name.rsplit("/", 1)[0]
                        if source not in best_per_source or score > best_per_source[source][0]:
                            best_per_source[source] = (score, name)

                if self.use_local_peak:
                    for source, (score, name) in best_per_source.items():
                        buf = peak_buf[source]
                        buf.append((frame_index, score, name))
                        if len(buf) >= 3:
                            _, mid_score, mid_name = buf[1]
                            if buf[0][1] <= buf[1][1] and buf[1][1] >= buf[2][1]:
                                self.result_item.emit(buf[1][0], mid_name, mid_score)
                            buf.pop(0)
                else:
                    matched_this_frame = set(best_per_source.keys())
                    for source, (score, name) in best_per_source.items():
                        if not prev_matched.get(source, False):
                            self.result_item.emit(frame_index, name, score)
                    for src in sources:
                        prev_matched[src] = src in matched_this_frame

                step_index += 1
                self.progress.emit(step_index, num_steps)

            if self.use_local_peak:
                for source, buf in peak_buf.items():
                    if len(buf) >= 1 and buf[-1][1] >= 0:
                        self.result_item.emit(buf[-1][0], buf[-1][2], buf[-1][1])
        finally:
            cap.release()
        self.finished_detect.emit()


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
        self.detect_thread: DetectThread | None = None
        self._detect_results: list[tuple[int, str, float]] = []
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
        video_area.setMinimumSize(540, 720)
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

        right_splitter = QSplitter(Qt.Horizontal)

        right_1 = QWidget()
        right_1.setMinimumWidth(200)
        right_layout = QVBoxLayout(right_1)
        right_layout.setContentsMargins(0, 0, 0, 0)
        detect_title = QLabel("テンプレート一致検出")
        detect_title.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(detect_title)
        detect_desc = QLabel("動画を開いた状態で「検出実行」を押すと、\ngo/frames・timeup/frames・result/frames の画像に近いフレームを検出します。")
        detect_desc.setWordWrap(True)
        detect_desc.setStyleSheet("color: #666; font-size: 11px;")
        right_layout.addWidget(detect_desc)
        _detect_settings = QSettings("analyzer", "main")
        detect_opts = QHBoxLayout()
        detect_opts.addWidget(QLabel("間隔:"))
        self.spin_frame_step = QSpinBox()
        self.spin_frame_step.setMinimum(1)
        self.spin_frame_step.setMaximum(300)
        self.spin_frame_step.setValue(_detect_settings.value("detectFrameStep", 5, type=int))
        self.spin_frame_step.setToolTip("何フレームおきに比較するか")
        detect_opts.addWidget(self.spin_frame_step)
        detect_opts.addWidget(QLabel("閾値:"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setMinimum(0.0)
        self.spin_threshold.setMaximum(1.0)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(float(_detect_settings.value("detectThreshold", 0.75, type=float)))
        self.spin_threshold.setToolTip("go/frames 用。この値以上を一致とみなす (0～1)")
        detect_opts.addWidget(self.spin_threshold)
        detect_opts.addWidget(QLabel("timeup 閾値:"))
        self.spin_threshold_timeup = QDoubleSpinBox()
        self.spin_threshold_timeup.setMinimum(0.0)
        self.spin_threshold_timeup.setMaximum(1.0)
        self.spin_threshold_timeup.setSingleStep(0.05)
        self.spin_threshold_timeup.setValue(float(_detect_settings.value("detectThresholdTimeup", 0.75, type=float)))
        self.spin_threshold_timeup.setToolTip("timeup/frames 用 (0～1)")
        detect_opts.addWidget(self.spin_threshold_timeup)
        detect_opts.addWidget(QLabel("result 閾値:"))
        self.spin_threshold_result = QDoubleSpinBox()
        self.spin_threshold_result.setMinimum(0.0)
        self.spin_threshold_result.setMaximum(1.0)
        self.spin_threshold_result.setSingleStep(0.05)
        self.spin_threshold_result.setValue(float(_detect_settings.value("detectThresholdResult", 0.75, type=float)))
        self.spin_threshold_result.setToolTip("result/frames 用 (0～1)")
        detect_opts.addWidget(self.spin_threshold_result)
        right_layout.addLayout(detect_opts)
        self.check_local_peak = QCheckBox("ローカル最大のみ（近似でスコアが違ってもピークなら検出）")
        self.check_local_peak.setChecked(_detect_settings.value("detectLocalPeak", False, type=bool))
        self.check_local_peak.setToolTip("オンにすると、閾値以上かつ「前後のフレームよりスコアが高い」ときだけ報告。シーンごとのスコアばらつきに強くなる")
        right_layout.addWidget(self.check_local_peak)
        self.btn_detect = QPushButton("検出実行")
        self.btn_detect.clicked.connect(self._run_detect)
        right_layout.addWidget(self.btn_detect)
        self.label_detect_progress = QLabel("")
        right_layout.addWidget(self.label_detect_progress)
        self.progress_detect = QProgressBar()
        self.progress_detect.setVisible(False)
        right_layout.addWidget(self.progress_detect)
        self.list_detect_results = QListWidget()
        self.list_detect_results.setToolTip("クリックでそのフレームへ移動")
        self.list_detect_results.itemClicked.connect(self._on_detect_result_clicked)
        right_layout.addWidget(self.list_detect_results, 1)
        right_splitter.addWidget(right_1)

        right_2 = QFrame()
        right_2.setFrameStyle(QFrame.StyledPanel)
        right_2.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        right_2.setMinimumWidth(120)
        right_2_layout = QVBoxLayout(right_2)
        right_2_layout.addWidget(QLabel("（エリア2）"))
        right_splitter.addWidget(right_2)

        right_splitter.setSizes([320, 280])
        right_splitter.setMinimumWidth(400)
        splitter.addWidget(right_splitter)

        splitter.setSizes([480, 440])
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
        detecting = self.detect_thread is not None and self.detect_thread.isRunning()
        self.btn_detect.setEnabled(has_video and not detecting)
        self.spin_frame_step.setEnabled(has_video and not detecting)
        self.spin_threshold.setEnabled(has_video and not detecting)
        self.spin_threshold_timeup.setEnabled(has_video and not detecting)
        self.spin_threshold_result.setEnabled(has_video and not detecting)
        self.check_local_peak.setEnabled(has_video and not detecting)

    def _run_detect(self):
        if not self.video_path or self.total_frames == 0:
            return
        templates = load_templates()
        if not templates:
            QMessageBox.warning(
                self,
                "検出",
                "テンプレートがありません。\n"
                + str(_get_templates_base_dir())
                + "\nの go/frames・timeup/frames・result/frames に PNG/JPEG 画像を置いてください。",
            )
            return
        self._detect_results.clear()
        self.list_detect_results.clear()
        step = self.spin_frame_step.value()
        threshold_go = self.spin_threshold.value()
        threshold_timeup = self.spin_threshold_timeup.value()
        threshold_result = self.spin_threshold_result.value()
        _s = QSettings("analyzer", "main")
        _s.setValue("detectFrameStep", step)
        _s.setValue("detectThreshold", threshold_go)
        _s.setValue("detectThresholdTimeup", threshold_timeup)
        _s.setValue("detectThresholdResult", threshold_result)
        use_local_peak = self.check_local_peak.isChecked()
        _s.setValue("detectLocalPeak", use_local_peak)
        templates = load_templates(
            threshold_go=threshold_go,
            threshold_timeup=threshold_timeup,
            threshold_result=threshold_result,
        )
        if not templates:
            return
        self.detect_thread = DetectThread(
            self.video_path, self.total_frames, self.fps, templates, step, use_local_peak
        )
        self.detect_thread.progress.connect(self._on_detect_progress)
        self.detect_thread.result_item.connect(self._on_detect_result_item)
        self.detect_thread.finished_detect.connect(self._on_detect_finished)
        self.progress_detect.setVisible(True)
        self.progress_detect.setMaximum(0)
        self.progress_detect.setValue(0)
        self.label_detect_progress.setText("検出中...")
        self.update_ui_state(True)
        self.detect_thread.start()

    def _on_detect_progress(self, current: int, total: int):
        self.progress_detect.setMaximum(total)
        self.progress_detect.setValue(current)
        self.label_detect_progress.setText(f"{current} / {total}")

    def _on_detect_result_item(self, frame_index: int, template_name: str, score: float):
        self._detect_results.append((frame_index, template_name, score))
        time_str = self._frame_to_time(frame_index)
        item = QListWidgetItem(f"{time_str}  #{frame_index}  {template_name}  ({score:.2f})")
        item.setData(Qt.ItemDataRole.UserRole, frame_index)
        self.list_detect_results.addItem(item)

    def _on_detect_finished(self):
        self.detect_thread = None
        self.progress_detect.setVisible(False)
        self.label_detect_progress.setText(f"検出完了（{len(self._detect_results)} 件）")
        self.update_ui_state(self.cap is not None)

    def _on_detect_result_clicked(self, item: QListWidgetItem):
        frame_index = item.data(Qt.ItemDataRole.UserRole)
        if frame_index is not None and self.cap is not None:
            self._show_frame_at(int(frame_index))

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
        self._stop_detect_thread()
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

    def _stop_detect_thread(self):
        if self.detect_thread and self.detect_thread.isRunning():
            self.detect_thread.stop()
            self.detect_thread.wait(5000)
        self.detect_thread = None

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
