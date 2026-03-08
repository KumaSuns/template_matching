"""
analyzer_dl: 動画を取り込み、学習済みモデルでフレームを分類・検出する
"""
import sys
from pathlib import Path

# ランチャーから起動しても model/classes を import できるようにする
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np  # OpenCV より先に import（検出実行時の「Numpy is not available」対策）
import cv2
from PySide6.QtCore import Qt, QSettings, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

try:
    import torch
    from model import build_model, IMAGENET_MEAN, IMAGENET_STD
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    torch = None
    build_model = None
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QSplitter,
    QFrame,
    QSizePolicy,
    QDialog,
    QLineEdit,
    QTextEdit,
    QDialogButtonBox,
)

def cv2_to_qpixmap(bgr_image, scale_w: int = 0, scale_h: int = 0):
    if bgr_image is None or bgr_image.size == 0:
        return QPixmap()
    h, w = bgr_image.shape[:2]
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) if len(bgr_image.shape) == 3 else cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2RGB)
    bytes_per_line = rgb.shape[2] * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg.copy())
    if scale_w > 0 and scale_h > 0 and (scale_w < w or scale_h < h):
        pix = pix.scaled(scale_w, scale_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


def load_dl_model(path: str):
    """model.pth を読み込み (model, classes_list, error_message) を返す。失敗時は None, [], エラー文"""
    if not _TORCH_AVAILABLE:
        return None, [], "torch がインストールされていません"
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            return None, [], "チェックポイントが辞書形式ではありません"
        classes_list = ckpt.get("classes", [])
        num_classes = ckpt.get("num_classes", len(classes_list))
        if not classes_list or num_classes <= 0:
            return None, [], "チェックポイントに classes がありません"
        if "model_state_dict" not in ckpt:
            return None, [], "チェックポイントに model_state_dict がありません"
        model = build_model(num_classes, pretrained=False)
        state_dict = ckpt["model_state_dict"]
        # DataParallel で保存した場合の "module." プレフィックスを外す
        if next(iter(state_dict.keys()), "").startswith("module."):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model, classes_list, None
    except Exception as e:
        return None, [], str(e)


class AddToDataDialog(QDialog):
    """検出結果のフレームをどのクラスで保存するか指定するダイアログ"""
    def __init__(self, parent, frame_index: int, detected_cls: str, score: float, classes: list, train_dir: str, val_dir: str):
        super().__init__(parent)
        self.setWindowTitle("データに追加")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"フレーム #{frame_index}（検出: {detected_cls} {score:.2f}）を、どのクラスとして保存しますか？"))
        self.combo_class = QComboBox()
        self.combo_class.addItems(classes)
        idx = self.combo_class.findText(detected_cls)
        if idx >= 0:
            self.combo_class.setCurrentIndex(idx)
        layout.addWidget(QLabel("クラス:"))
        layout.addWidget(self.combo_class)
        self.combo_subset = QComboBox()
        self.combo_subset.addItem("train", False)
        self.combo_subset.addItem("val", True)
        layout.addWidget(QLabel("保存先:"))
        layout.addWidget(self.combo_subset)
        layout.addWidget(QLabel(f"  train → {train_dir}\n  val   → {val_dir}"))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def get_selected_class(self) -> str:
        return self.combo_class.currentText()

    def get_use_val(self) -> bool:
        return self.combo_subset.currentData()


class TrainThread(QThread):
    """学習を別スレッドで実行。progress(epoch, total, train_loss, val_acc) val_acc=-1 は未使用"""
    progress = Signal(int, int, float, float)
    finished_with_error = Signal(str)
    finished_ok = Signal(str)

    def __init__(self, data_dir: str, epochs: int, out_path: str):
        super().__init__()
        self.data_dir = data_dir
        self.epochs = epochs
        self.out_path = out_path
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        if not _TORCH_AVAILABLE:
            self.finished_with_error.emit("torch がインストールされていません")
            return
        try:
            from train import run_training

            def on_progress(epoch, total, train_loss, val_acc):
                self.progress.emit(epoch, total, train_loss, val_acc if val_acc is not None else -1.0)
                return self._cancel

            err = run_training(
                self.data_dir,
                self.epochs,
                self.out_path,
                progress_callback=on_progress,
            )
            if err:
                self.finished_with_error.emit(err)
            else:
                self.finished_ok.emit(self.out_path)
        except Exception as e:
            self.finished_with_error.emit(str(e))


class AccuracyCheckThread(QThread):
    """精度チェックを別スレッドで実行"""
    finished = Signal(bool, str)

    def __init__(self, model, classes: list, data_dir: str):
        super().__init__()
        self.model = model
        self.classes = classes
        self.data_dir = data_dir

    def run(self):
        try:
            from check_model import get_accuracy_result
            ok, msg = get_accuracy_result(self.model, self.classes, Path(self.data_dir))
            self.finished.emit(ok, msg)
        except Exception as e:
            self.finished.emit(False, str(e))


class TrainDialog(QDialog):
    """学習の設定と実行ダイアログ"""

    def __init__(self, parent=None, last_data_dir: str = "", last_model_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("学習")
        self.setMinimumWidth(480)
        self._thread = None
        self._out_path = ""

        layout = QVBoxLayout(self)
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("データフォルダ:"))
        self.edit_data_dir = QLineEdit()
        self.edit_data_dir.setPlaceholderText("data/train があるフォルダ")
        self.edit_data_dir.setText(last_data_dir or str(Path(__file__).resolve().parent / "data"))
        data_row.addWidget(self.edit_data_dir, 1)
        btn_data = QPushButton("参照...")
        btn_data.clicked.connect(self._browse_data)
        data_row.addWidget(btn_data)
        layout.addLayout(data_row)

        epochs_row = QHBoxLayout()
        epochs_row.addWidget(QLabel("エポック数:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setMinimum(1)
        self.spin_epochs.setMaximum(500)
        self.spin_epochs.setValue(30)
        epochs_row.addWidget(self.spin_epochs)
        epochs_row.addStretch()
        layout.addLayout(epochs_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("保存先:"))
        self.edit_out = QLineEdit()
        self.edit_out.setPlaceholderText("model.pth")
        self.edit_out.setText(str(Path(last_model_dir) / "model.pth") if last_model_dir else str(Path(__file__).resolve().parent / "model.pth"))
        out_row.addWidget(self.edit_out, 1)
        btn_out = QPushButton("参照...")
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(btn_out)
        layout.addLayout(out_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        layout.addWidget(self.log_text)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("学習開始")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("閉じる")
        self.buttons.accepted.connect(self._start_train)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _browse_data(self):
        d = QFileDialog.getExistingDirectory(self, "データフォルダ", self.edit_data_dir.text())
        if d:
            self.edit_data_dir.setText(d)

    def _browse_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "モデル保存先", self.edit_out.text(), "PyTorch (*.pth);;All (*.*)")
        if path:
            self.edit_out.setText(path)

    def _start_train(self):
        data_dir = self.edit_data_dir.text().strip()
        out_path = self.edit_out.text().strip()
        if not data_dir:
            QMessageBox.warning(self, "学習", "データフォルダを指定してください。")
            return
        if not out_path:
            QMessageBox.warning(self, "学習", "保存先を指定してください。")
            return
        train_dir = Path(data_dir) / "train"
        if not train_dir.is_dir():
            QMessageBox.warning(self, "学習", f"{train_dir} がありません。\ndata/train/<クラス名>/ に画像を入れてください。")
            return

        self.buttons.setEnabled(False)
        self.edit_data_dir.setEnabled(False)
        self.edit_out.setEnabled(False)
        self.spin_epochs.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)
        self.log_text.clear()
        self.log_text.append("学習を開始しています...")

        self._thread = TrainThread(data_dir, self.spin_epochs.value(), out_path)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_ok.connect(self._on_finished_ok)
        self._thread.finished_with_error.connect(self._on_finished_error)
        self._thread.start()

    def _on_progress(self, epoch: int, total: int, train_loss: float, val_acc: float):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(epoch)
        msg = f"Epoch {epoch}/{total}  train_loss={train_loss:.4f}"
        if val_acc >= 0:
            msg += f"  val_acc={val_acc:.4f}"
        self.log_text.append(msg)

    def _on_finished_ok(self, out_path: str):
        self._out_path = out_path
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.log_text.append(f"完了。保存先: {out_path}")
        self.buttons.setEnabled(True)
        self.edit_data_dir.setEnabled(True)
        self.edit_out.setEnabled(True)
        self.spin_epochs.setEnabled(True)
        self._thread = None
        if self.parent() and hasattr(self.parent(), "load_model_path"):
            if QMessageBox.question(
                self, "学習完了", "このモデルを読み込みますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            ) == QMessageBox.StandardButton.Yes:
                self.parent().load_model_path(out_path)

    def _on_finished_error(self, msg: str):
        self.log_text.append(f"エラー: {msg}")
        QMessageBox.critical(self, "学習", msg)
        self.buttons.setEnabled(True)
        self.edit_data_dir.setEnabled(True)
        self.edit_out.setEnabled(True)
        self.spin_epochs.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._thread = None


class AnalyzerDLWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.video_path = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.fps = 30.0
        self._current_frame_np = None
        self.dl_model = None
        self.dl_classes = []
        self._dl_model_path = ""
        self._dl_cap = None
        self._dl_timer = QTimer(self)
        self._dl_timer.setSingleShot(True)
        self._dl_timer.timeout.connect(self._dl_timer_tick)
        self._dl_step_index = 0
        self._dl_num_steps = 0
        self._dl_prev_cls = "none"
        self._settings = QSettings("analyzer_dl", "main")
        self._last_video_dir = self._settings.value("lastVideoDir", "", type=str)
        self._last_model_dir = self._settings.value("lastModelDir", str(Path(__file__).resolve().parent), type=str)
        self._last_export_dir = self._settings.value("lastExportDir", "", type=str) or str(Path(__file__).resolve().parent)
        _mp = self._settings.value("modelPaths", [])
        if isinstance(_mp, list):
            _model_paths_raw = _mp
        elif isinstance(_mp, str) and _mp:
            _model_paths_raw = [s.strip() for s in _mp.split("|") if s.strip()]
        else:
            _model_paths_raw = []
        self._model_paths = [str(p) for p in _model_paths_raw[:15] if p and Path(str(p)).is_file()]
        self.setWindowTitle("analyzer (DL)")
        self.setMinimumSize(900, 600)
        self.setup_ui()
        last_path = self._settings.value("lastModelPath", "", type=str)
        if last_path and Path(last_path).is_file():
            self.load_model_path(last_path)
        else:
            self._update_model_status("")

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setSpacing(8)
        main.setContentsMargins(8, 8, 8, 8)

        open_row = QHBoxLayout()
        self.btn_open = QPushButton("動画を開く")
        self.label_path = QLabel("（未選択）")
        self.label_path.setStyleSheet("color: #666;")
        open_row.addWidget(self.btn_open)
        open_row.addWidget(self.label_path, 1)
        main.addLayout(open_row)
        self.btn_open.clicked.connect(self.open_video)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left.setMinimumWidth(400)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.StyledPanel)
        video_frame.setStyleSheet("QFrame { border: 1px solid #ddd; }")
        video_frame.setMinimumSize(560, 420)
        vl = QVBoxLayout(video_frame)
        self.label_video = QLabel()
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setText("動画を開いてください")
        self.label_video.setStyleSheet("color: #999;")
        self.label_video.setScaledContents(False)
        self.label_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_video.setMinimumSize(320, 240)
        vl.addWidget(self.label_video)
        left_layout.addWidget(video_frame)

        seek_row = QHBoxLayout()
        self.label_time = QLabel("0:00 / 0:00")
        self.label_time.setMinimumWidth(80)
        self.slider_seek = QSlider(Qt.Horizontal)
        self.slider_seek.setMinimum(0)
        self.slider_seek.setMaximum(0)
        self.slider_seek.valueChanged.connect(self.on_seek)
        seek_row.addWidget(self.label_time)
        seek_row.addWidget(self.slider_seek, 1)
        left_layout.addLayout(seek_row)
        step_row = QHBoxLayout()
        step_row.addStretch()
        self.step_buttons = []
        for label, delta in [("−30", -30), ("−10", -10), ("−5", -5), ("−1", -1), ("+1", 1), ("+5", 5), ("+10", 10), ("+30", 30)]:
            btn = QPushButton(label)
            btn.setFixedWidth(44)
            btn.clicked.connect(lambda checked=False, d=delta: self._frame_step(d))
            step_row.addWidget(btn)
            self.step_buttons.append(btn)
        step_row.addStretch()
        left_layout.addLayout(step_row)
        splitter.addWidget(left)

        right = QWidget()
        right.setMinimumWidth(260)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("検出"))
        desc = QLabel("学習済みモデル（model.pth）でフレームを分類し、go/timeup/result などを検出します。")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 11px;")
        right_layout.addWidget(desc)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("モデル:"))
        self.combo_models = QComboBox()
        self.combo_models.setMinimumWidth(120)
        self._refresh_model_combo()
        self.combo_models.activated.connect(self._on_model_combo_activated)
        model_row.addWidget(self.combo_models, 1)
        self.btn_add_model = QPushButton("追加...")
        self.btn_add_model.setToolTip("model.pth を開いてリストに追加し、読み込みます")
        self.btn_add_model.clicked.connect(self._open_add_model_dialog)
        model_row.addWidget(self.btn_add_model)
        right_layout.addLayout(model_row)
        self.label_model_status = QLabel("（未読み込み）")
        self.label_model_status.setStyleSheet("color: #666; font-size: 10px;")
        self.label_model_status.setWordWrap(True)
        right_layout.addWidget(self.label_model_status)
        self.btn_train = QPushButton("学習...")
        self.btn_train.clicked.connect(self._open_train_dialog)
        right_layout.addWidget(self.btn_train)
        self.btn_accuracy_check = QPushButton("精度チェック")
        self.btn_accuracy_check.clicked.connect(self._run_accuracy_check)
        if not _TORCH_AVAILABLE:
            self.btn_accuracy_check.setToolTip("torch がインストールされていません")
        right_layout.addWidget(self.btn_accuracy_check)
        opts = QHBoxLayout()
        opts.addWidget(QLabel("間隔:"))
        self.spin_step = QSpinBox()
        self.spin_step.setMinimum(1)
        self.spin_step.setMaximum(300)
        self.spin_step.setValue(self._settings.value("detectFrameStep", 5, type=int))
        self.spin_step.valueChanged.connect(self._save_step_setting)
        opts.addWidget(self.spin_step)
        right_layout.addLayout(opts)
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("信頼度:"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setMinimum(0.0)
        self.spin_conf.setMaximum(1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(self._settings.value("detectConfThreshold", 0.3, type=float))
        self.spin_conf.valueChanged.connect(self._save_conf_setting)
        self.spin_conf.setToolTip("この値以上の信頼度の予測だけを検出として表示します（精度向上用）")
        conf_row.addWidget(self.spin_conf)
        right_layout.addLayout(conf_row)
        margin_row = QHBoxLayout()
        margin_row.addWidget(QLabel("noneとの差:"))
        self.spin_none_margin = QDoubleSpinBox()
        self.spin_none_margin.setMinimum(0.0)
        self.spin_none_margin.setMaximum(1.0)
        self.spin_none_margin.setSingleStep(0.05)
        self.spin_none_margin.setDecimals(2)
        self.spin_none_margin.setValue(self._settings.value("detectNoneMargin", 0.05, type=float))
        self.spin_none_margin.valueChanged.connect(self._save_none_margin_setting)
        self.spin_none_margin.setToolTip("go/timeup/result は、none の信頼度よりこの値以上高いときだけ検出。0=オフ。go が一枚も出ないときは 0 に、none が go と出るなら 0.15〜0.25 に")
        margin_row.addWidget(self.spin_none_margin)
        right_layout.addLayout(margin_row)
        self.btn_detect_dl = QPushButton("検出実行")
        self.btn_detect_dl.clicked.connect(self.run_detect_dl)
        if not _TORCH_AVAILABLE:
            self.btn_detect_dl.setToolTip("torch がインストールされていません")
        right_layout.addWidget(self.btn_detect_dl)
        self.btn_predict_current = QPushButton("現在フレームを判定")
        self.btn_predict_current.clicked.connect(self.predict_current_frame)
        if not _TORCH_AVAILABLE:
            self.btn_predict_current.setToolTip("torch がインストールされていません")
        right_layout.addWidget(self.btn_predict_current)
        self.label_progress = QLabel("")
        right_layout.addWidget(self.label_progress)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        self.list_results = QListWidget()
        self.list_results.itemClicked.connect(self.on_result_clicked)
        self.list_results.itemSelectionChanged.connect(self._on_result_selection_changed)
        right_layout.addWidget(self.list_results, 1)
        btn_row = QHBoxLayout()
        self.btn_add_to_data = QPushButton("データに追加")
        self.btn_add_to_data.setToolTip("選択した検出結果のフレームを、指定クラスで学習用データに保存します")
        self.btn_add_to_data.clicked.connect(self.add_selected_result_to_data)
        btn_row.addWidget(self.btn_add_to_data)
        self.btn_export = QPushButton("結果を保存（CSV）")
        self.btn_export.clicked.connect(self.export_results_csv)
        btn_row.addWidget(self.btn_export)
        right_layout.addLayout(btn_row)
        splitter.addWidget(right)

        splitter.setSizes([500, 320])
        main.addWidget(splitter)
        self.update_ui_state(False)

    def _save_step_setting(self, value: int):
        self._settings.setValue("detectFrameStep", value)

    def _save_conf_setting(self, value: float):
        self._settings.setValue("detectConfThreshold", value)

    def _save_none_margin_setting(self, value: float):
        self._settings.setValue("detectNoneMargin", value)

    def _refresh_model_combo(self):
        self.combo_models.blockSignals(True)
        self.combo_models.clear()
        for p in self._model_paths:
            self.combo_models.addItem(Path(p).name, p)
        self.combo_models.addItem("追加...", None)
        idx = -1
        if self._dl_model_path:
            for i, p in enumerate(self._model_paths):
                if p == self._dl_model_path:
                    idx = i
                    break
        if idx >= 0:
            self.combo_models.setCurrentIndex(idx)
        else:
            self.combo_models.setCurrentIndex(max(0, self.combo_models.count() - 1))
        self.combo_models.setToolTip(self._dl_model_path or "")
        self.combo_models.blockSignals(False)

    def _open_add_model_dialog(self):
        """モデルを追加するためのファイルダイアログを開く"""
        path, _ = QFileDialog.getOpenFileName(
            self, "モデルを追加", self._last_model_dir,
            "PyTorch (*.pth);;All (*.*)"
        )
        if path:
            self._last_model_dir = str(Path(path).parent)
            self._settings.setValue("lastModelDir", self._last_model_dir)
            self.load_model_path(path)

    def _on_model_combo_activated(self, index: int):
        if index < 0 or index >= self.combo_models.count():
            return
        path = self.combo_models.itemData(index)
        if path is None:
            self._open_add_model_dialog()
        else:
            self.load_model_path(path)

    def update_ui_state(self, has_video: bool):
        busy = self._dl_cap is not None
        check_busy = getattr(self, "_accuracy_thread", None) is not None
        self.btn_detect_dl.setEnabled(has_video and not busy and _TORCH_AVAILABLE and bool(self._dl_model_path))
        self.btn_predict_current.setEnabled(has_video and not busy and _TORCH_AVAILABLE and bool(self._dl_model_path))
        self.btn_accuracy_check.setEnabled(not busy and not check_busy and _TORCH_AVAILABLE and bool(self._dl_model_path))
        self.combo_models.setEnabled(not busy)
        self.btn_add_model.setEnabled(not busy)
        has_sel = self.list_results.currentItem() is not None
        self.btn_add_to_data.setEnabled(not busy and has_video and has_sel)
        self.slider_seek.setEnabled(has_video)
        self.spin_step.setEnabled(has_video)
        for btn in getattr(self, "step_buttons", []):
            btn.setEnabled(has_video)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "動画を開く", self._last_video_dir,
            "Video (*.mp4 *.avi *.mov *.mkv *.wmv *.webm);;All (*.*)"
        )
        if not path:
            return
        self._last_video_dir = str(Path(path).parent)
        self._settings.setValue("lastVideoDir", self._last_video_dir)
        if self._dl_cap is not None:
            self._dl_timer.stop()
            self._dl_cap.release()
            self._dl_cap = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "エラー", "動画を開けませんでした。")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_path = path
        self.current_frame_index = 0
        self.label_path.setText(Path(path).name)
        self.label_path.setToolTip(path)
        self.slider_seek.setMaximum(max(0, self.total_frames - 1))
        self.update_ui_state(True)
        self._show_frame_at(0)
        self._update_time_label()

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

    def _show_frame_at(self, frame_index: int):
        if self.cap is None or self.total_frames == 0:
            return
        frame_index = max(0, min(frame_index, self.total_frames - 1))
        self.current_frame_index = frame_index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self._current_frame_np = frame.copy()
            w = min(self.label_video.width(), 1920)
            h = min(self.label_video.height(), 1080)
            if w > 0 and h > 0:
                pix = cv2_to_qpixmap(self._current_frame_np, w, h)
                self.label_video.setPixmap(pix)
                self.label_video.setStyleSheet("")
        self._update_time_label()

    def on_seek(self, value: int):
        self._show_frame_at(value)

    def _frame_step(self, delta: int):
        """コマ送り: delta フレームだけ進める／戻す"""
        if self.cap is None or self.total_frames == 0:
            return
        self._show_frame_at(max(0, min(self.current_frame_index + delta, self.total_frames - 1)))

    def load_model_path(self, path: str):
        """指定パスのモデルを読み込む（ファイルダイアログなし）"""
        if not path or not Path(path).is_file():
            return
        model, classes_list, err = load_dl_model(path)
        if model is None or not classes_list:
            msg = "モデルを読み込めませんでした。"
            if err:
                msg += f"\n\n{err}"
            QMessageBox.warning(self, "モデル", msg)
            return
        self._settings.setValue("lastModelPath", path)
        path_str = str(path)
        if path_str not in self._model_paths:
            self._model_paths.insert(0, path_str)
            self._model_paths = self._model_paths[:15]
            self._settings.setValue("modelPaths", "|".join(self._model_paths))
        self._dl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dl_model = model.to(self._dl_device)
        self.dl_classes = classes_list
        self._dl_model_path = path_str
        self._norm_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=self._dl_device).view(1, 3, 1, 1)
        self._norm_std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=self._dl_device).view(1, 3, 1, 1)
        self._refresh_model_combo()
        self.combo_models.setToolTip(path_str)
        self._update_model_status(path_str)
        self.update_ui_state(self.cap is not None)

    def _update_model_status(self, path: str = ""):
        if not path or not Path(path).is_file():
            self.label_model_status.setText("（未読み込み）")
            return
        lines = [f"使用中: {Path(path).name}"]
        try:
            mtime = Path(path).stat().st_mtime
            from datetime import datetime
            dt = datetime.fromtimestamp(mtime)
            lines.append(f"更新: {dt.strftime('%Y-%m-%d %H:%M')}")
        except Exception:
            pass
        if getattr(self, "dl_classes", None):
            # モデル内のクラス順（0=none かどうか確認用）
            lines.append("クラス(0から): " + ", ".join(f"{i}:{c}" for i, c in enumerate(self.dl_classes)))
        if getattr(self, "_dl_device", None):
            lines.append(f"推論: {self._dl_device.type}")
        self.label_model_status.setText("\n".join(lines))

    def _run_accuracy_check(self):
        if not self.dl_model or not self.dl_classes:
            QMessageBox.warning(self, "精度チェック", "先に「追加...」で model.pth を読み込んでください。")
            return
        data_dir = _script_dir / "data"
        if not data_dir.is_dir():
            QMessageBox.warning(self, "精度チェック", f"データフォルダがありません:\n{data_dir}")
            return
        self.btn_accuracy_check.setEnabled(False)
        self._accuracy_thread = AccuracyCheckThread(self.dl_model, self.dl_classes, str(data_dir))
        self._accuracy_thread.finished.connect(self._on_accuracy_check_done)
        self._accuracy_thread.start()

    def _on_accuracy_check_done(self, ok: bool, msg: str):
        self._accuracy_thread = None
        self.btn_accuracy_check.setEnabled(True)
        if ok:
            QMessageBox.information(self, "精度チェック", msg)
        else:
            QMessageBox.warning(self, "精度チェック", msg)

    def _open_train_dialog(self):
        last_data = self._settings.value("lastDataDir", str(Path(__file__).resolve().parent / "data"), type=str)
        d = TrainDialog(self, last_data, self._last_model_dir)
        d.exec()
        if d.edit_data_dir.text():
            self._settings.setValue("lastDataDir", str(Path(d.edit_data_dir.text()).resolve()))
        if d.edit_out.text():
            self._settings.setValue("lastModelDir", str(Path(d.edit_out.text()).resolve().parent))

    def run_detect_dl(self):
        if not self.video_path or self.total_frames == 0:
            return
        if not self.dl_model or not self.dl_classes:
            QMessageBox.warning(self, "DL検出", "先に「追加...」で model.pth を読み込んでください。")
            return
        step = max(1, self.spin_step.value())
        self._settings.setValue("detectFrameStep", step)
        self.list_results.clear()
        self.progress_bar.setVisible(True)
        self._dl_num_steps = (self.total_frames + step - 1) // step
        self.progress_bar.setMaximum(self._dl_num_steps)
        self.progress_bar.setValue(0)
        self.label_progress.setText("DL検出中...")
        self.update_ui_state(True)
        self._dl_cap = cv2.VideoCapture(self.video_path)
        if not self._dl_cap.isOpened():
            self._dl_cap = None
            self.progress_bar.setVisible(False)
            self.update_ui_state(self.cap is not None)
            QMessageBox.critical(self, "DL検出", "動画を開けませんでした。")
            return
        # OpenCV と NumPy の互換性チェック（検出実行時に「Numpy is not available」が出るのを事前検出）
        try:
            self._dl_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._dl_cap.read()
            if ret and frame is not None:
                _ = np.asarray(frame, dtype=np.uint8)
                torch.from_numpy(_).numel()
        except Exception as e:
            err_msg = str(e).lower()
            self._dl_cap.release()
            self._dl_cap = None
            self.progress_bar.setVisible(False)
            self.update_ui_state(self.cap is not None)
            err_path = _script_dir / "last_error.txt"
            try:
                err_path.write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
            except Exception:
                pass
            if "numpy" in err_msg or "array" in err_msg or "multiarray" in err_msg or "numpy is not available" in err_msg:
                QMessageBox.critical(
                    self, "DL検出",
                    "「Numpy is not available」が発生しました。\n\n"
                    "ターミナルで次を実行してください（opencv-python の代わりに headless 版を試します）:\n\n"
                    "  .venv/bin/pip uninstall opencv-python opencv-python-headless -y\n"
                    "  .venv/bin/pip install \"numpy>=1.24,<2\"\n"
                    "  .venv/bin/pip install opencv-python-headless\n\n"
                    "それでも出る場合は、opencv-python に戻して:\n"
                    "  .venv/bin/pip uninstall opencv-python-headless -y\n"
                    "  .venv/bin/pip install opencv-python\n\n"
                    f"詳細は {err_path.name} に保存しました。"
                )
            else:
                QMessageBox.critical(
                    self, "DL検出",
                    f"検出の準備中にエラーが発生しました。\n詳細は {err_path.name} に保存しました。"
                )
            return
        self._dl_step_index = 0
        self._dl_prev_cls = "none"
        self._dl_timer.start(0)

    def predict_current_frame(self):
        """表示中のフレームを1枚だけ判定し、結果を表示する（検知されない原因の確認用）"""
        if not self.cap or not self.dl_model or not self.dl_classes:
            QMessageBox.warning(self, "判定", "動画とモデルを読み込んでください。")
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            QMessageBox.warning(self, "判定", "フレームを読み込めませんでした。")
            return
        try:
            img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = np.asarray(img, dtype=np.uint8)
            x = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self._dl_device)
            x = (x - self._norm_mean) / self._norm_std
            with torch.no_grad():
                logits = self.dl_model(x)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(logits.argmax(dim=1).item())
            cls_name = self.dl_classes[pred_idx] if pred_idx < len(self.dl_classes) else "?"
            conf = float(probs[0, pred_idx].item())
            lines = [f"予測: {cls_name}  (信頼度 {conf:.2%})", f"フレーム #{self.current_frame_index}  {self._frame_to_time(self.current_frame_index)}"]
            lines.append("")
            lines.append("各クラスの信頼度:")
            for i, name in enumerate(self.dl_classes):
                if i < probs.shape[1]:
                    lines.append(f"  {name}: {probs[0, i].item():.2%}")
            QMessageBox.information(self, "現在フレームの判定", "\n".join(lines))
        except Exception as e:
            err_path = _script_dir / "last_error.txt"
            try:
                err_path.write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
            except Exception:
                pass
            QMessageBox.critical(self, "判定エラー", f"詳細は {err_path.name} に保存しました。")

    def _dl_timer_tick(self):
        if self._dl_cap is None:
            return
        step = max(1, self.spin_step.value())
        frame_index = self._dl_step_index * step
        if frame_index >= self.total_frames:
            self._dl_finish()
            return
        try:
            self._dl_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self._dl_cap.read()
            if not ret or frame is None:
                self._dl_step_index += 1
                self._on_progress(self._dl_step_index, self._dl_num_steps)
                self._dl_timer.start(0)
                return
            img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = np.asarray(img, dtype=np.uint8)
            x = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self._dl_device)
            x = (x - self._norm_mean) / self._norm_std
            with torch.no_grad():
                logits = self.dl_model(x)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(logits.argmax(dim=1).item())
                conf = float(probs[0, pred_idx].item())
            cls_name = self.dl_classes[pred_idx] if pred_idx < len(self.dl_classes) else "?"
            conf_threshold = self.spin_conf.value()
            margin = self.spin_none_margin.value()
            none_idx = next((i for i, c in enumerate(self.dl_classes) if c == "none"), None)
            prob_none = float(probs[0, none_idx].item()) if none_idx is not None and none_idx < probs.shape[1] else 0.0
            above_none = (conf - prob_none) >= margin
            if cls_name != "none" and conf >= conf_threshold and above_none and (self._dl_prev_cls != cls_name or self._dl_prev_cls == "none"):
                self._on_result_item(frame_index, cls_name, conf)
            self._dl_prev_cls = cls_name
        except Exception as e:
            self._dl_finish()
            err_path = _script_dir / "last_error.txt"
            try:
                err_path.write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
            except Exception:
                pass
            QMessageBox.critical(self, "DL検出エラー", f"詳細は {err_path.name} に保存しました。")
            return
        self._dl_step_index += 1
        self._on_progress(self._dl_step_index, self._dl_num_steps)
        if self._dl_step_index >= self._dl_num_steps:
            self._dl_finish()
        else:
            self._dl_timer.start(0)

    def _dl_finish(self):
        if self._dl_cap is not None:
            self._dl_cap.release()
            self._dl_cap = None
        self._dl_timer.stop()
        self.progress_bar.setVisible(False)
        self.label_progress.setText(f"検出完了（{self.list_results.count()} 件）")
        self.update_ui_state(self.cap is not None)

    def _on_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.label_progress.setText(f"{current} / {total}")

    def _on_result_item(self, frame_index: int, name: str, score: float):
        time_str = self._frame_to_time(frame_index)
        item = QListWidgetItem(f"{time_str}  #{frame_index}  {name}  ({score:.2f})")
        item.setData(Qt.ItemDataRole.UserRole, (frame_index, name, score))
        self.list_results.addItem(item)

    def on_result_clicked(self, item: QListWidgetItem):
        data = item.data(Qt.ItemDataRole.UserRole)
        if data is None or self.cap is None:
            return
        frame_index = data[0] if isinstance(data, tuple) else data
        self._show_frame_at(int(frame_index))

    def _on_result_selection_changed(self):
        self.update_ui_state(self.cap is not None)

    def add_selected_result_to_data(self):
        """選択した検出結果のフレームを、指定クラスで data/train または data/val に保存する"""
        item = self.list_results.currentItem()
        if item is None:
            QMessageBox.warning(self, "データに追加", "検出結果の一覧で、追加したい行を選択してください。")
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, tuple) or len(data) != 3:
            return
        frame_index, detected_cls, score = data
        if not self.video_path or not Path(self.video_path).is_file():
            QMessageBox.warning(self, "データに追加", "動画が開かれていません。")
            return
        if not self.dl_classes:
            QMessageBox.warning(self, "データに追加", "モデルが読み込まれていません。")
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "データに追加", "動画を開けませんでした。")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            QMessageBox.warning(self, "データに追加", "フレームを読み込めませんでした。")
            return
        data_dir = _script_dir / "data"
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        dialog = AddToDataDialog(
            self, frame_index, detected_cls, score, list(self.dl_classes),
            str(train_dir), str(val_dir)
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        class_name = dialog.get_selected_class()
        use_val = dialog.get_use_val()
        if not class_name:
            return
        dest_dir = (val_dir if use_val else train_dir) / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        base = dest_dir / f"frame_{frame_index:06d}"
        path = base.with_suffix(".png")
        n = 0
        while path.exists():
            n += 1
            path = base.parent / f"{base.stem}_{n}.png"
        try:
            cv2.imwrite(str(path), frame)
            QMessageBox.information(
                self, "データに追加",
                f"保存しました:\n{path}\n\nクラス: {class_name}\n{'val' if use_val else 'train'}"
            )
        except Exception as e:
            QMessageBox.critical(self, "データに追加", f"保存に失敗しました:\n{e}")

    def export_results_csv(self):
        if self.list_results.count() == 0:
            QMessageBox.information(self, "結果を保存", "検出結果がありません。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "結果を CSV に保存", self._last_export_dir,
            "CSV (*.csv);;All (*.*)"
        )
        if not path:
            return
        self._last_export_dir = str(Path(path).parent)
        self._settings.setValue("lastExportDir", self._last_export_dir)
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "time", "class", "confidence"])
            if self.video_path:
                w.writerow(["# video", Path(self.video_path).name])
            for i in range(self.list_results.count()):
                item = self.list_results.item(i)
                data = item.data(Qt.ItemDataRole.UserRole)
                if not isinstance(data, tuple) or len(data) != 3:
                    continue
                frame_index, name, score = data
                time_str = self._frame_to_time(frame_index)
                w.writerow([frame_index, time_str, name, f"{score:.4f}"])
        QMessageBox.information(self, "結果を保存", f"保存しました:\n{path}")

    def closeEvent(self, event):
        if self._dl_cap is not None:
            self._dl_timer.stop()
            self._dl_cap.release()
            self._dl_cap = None
        if self.cap:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = AnalyzerDLWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
