"""
テンプレートマッチングの PySide6 GUI
"""
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QSpinBox,
    QScrollArea,
    QMessageBox,
    QSplitter,
)

from match import match_template, draw_matches


def cv2_to_qpixmap(bgr_image):
    """OpenCV BGR 画像を QPixmap に変換"""
    if bgr_image is None or bgr_image.size == 0:
        return QPixmap()
    h, w = bgr_image.shape[:2]
    if len(bgr_image.shape) == 2:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    bytes_per_line = rgb.shape[2] * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class TemplateMatchingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path: str | None = None
        self.template_path: str | None = None
        self.setWindowTitle("テンプレートマッチング")
        self.setMinimumSize(900, 600)
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ファイル選択
        file_group = QGroupBox("ファイル")
        file_layout = QHBoxLayout(file_group)
        self.btn_image = QPushButton("検索対象画像を開く")
        self.btn_template = QPushButton("テンプレート画像を開く")
        self.label_image = QLabel("（未選択）")
        self.label_template = QLabel("（未選択）")
        self.label_image.setStyleSheet("color: #666;")
        self.label_template.setStyleSheet("color: #666;")
        file_layout.addWidget(self.btn_image)
        file_layout.addWidget(self.label_image)
        file_layout.addStretch()
        file_layout.addWidget(self.btn_template)
        file_layout.addWidget(self.label_template)
        layout.addWidget(file_group)

        self.btn_image.clicked.connect(self.select_image)
        self.btn_template.clicked.connect(self.select_template)

        # パラメータ
        param_group = QGroupBox("パラメータ")
        param_layout = QHBoxLayout(param_group)
        param_layout.addWidget(QLabel("手法:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems([
            "TM_CCOEFF_NORMED",
            "TM_CCORR_NORMED",
            "TM_SQDIFF_NORMED",
        ])
        param_layout.addWidget(self.combo_method)
        param_layout.addWidget(QLabel("閾値:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(80)
        self.slider_threshold.setTickPosition(QSlider.TicksBelow)
        self.slider_threshold.setTickInterval(10)
        param_layout.addWidget(self.slider_threshold)
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 100)
        self.spin_threshold.setValue(80)
        self.spin_threshold.setSuffix(" %")
        param_layout.addWidget(self.spin_threshold)
        self.slider_threshold.valueChanged.connect(self.spin_threshold.setValue)
        self.spin_threshold.valueChanged.connect(self.slider_threshold.setValue)
        param_layout.addStretch()
        layout.addWidget(param_group)

        # 実行
        self.btn_run = QPushButton("マッチング実行")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.clicked.connect(self.run_matching)
        layout.addWidget(self.btn_run)

        # 結果
        result_layout = QHBoxLayout()
        self.label_match_count = QLabel("マッチ数: —")
        self.label_match_count.setFont(QFont(None, 10))
        result_layout.addWidget(self.label_match_count)
        result_layout.addStretch()
        layout.addLayout(result_layout)

        splitter = QSplitter(Qt.Horizontal)
        self.label_result = QLabel()
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setMinimumSize(400, 300)
        self.label_result.setStyleSheet("background: #2d2d2d; color: #888;")
        self.label_result.setText("画像とテンプレートを選択して「マッチング実行」")
        scroll_result = QScrollArea()
        scroll_result.setWidget(self.label_result)
        scroll_result.setWidgetResizable(True)
        splitter.addWidget(scroll_result)
        layout.addWidget(splitter)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "検索対象画像", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.image_path = path
            self.label_image.setText(Path(path).name)

    def select_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "テンプレート画像", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.template_path = path
            self.label_template.setText(Path(path).name)

    def run_matching(self):
        if not self.image_path or not self.template_path:
            QMessageBox.warning(
                self, "未選択", "検索対象画像とテンプレート画像の両方を選択してください。"
            )
            return
        method = self.combo_method.currentText()
        threshold = self.slider_threshold.value() / 100.0
        try:
            points, size, _ = match_template(
                self.image_path, self.template_path, method=method, threshold=threshold
            )
        except FileNotFoundError as e:
            QMessageBox.critical(self, "エラー", str(e))
            return
        self.label_match_count.setText(f"マッチ数: {len(points)}")
        if points:
            img = draw_matches(self.image_path, points, size)
            pix = cv2_to_qpixmap(img)
            scale = min(800 / pix.width(), 600 / pix.height(), 1.0)
            if scale < 1.0:
                pix = pix.scaled(
                    int(pix.width() * scale), int(pix.height() * scale),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            self.label_result.setPixmap(pix)
            self.label_result.setStyleSheet("")
        else:
            img = cv2.imread(self.image_path)
            pix = cv2_to_qpixmap(img)
            scale = min(800 / pix.width(), 600 / pix.height(), 1.0)
            if scale < 1.0:
                pix = pix.scaled(
                    int(pix.width() * scale), int(pix.height() * scale),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            self.label_result.setPixmap(pix)
            self.label_result.setStyleSheet("")
            self.label_match_count.setText("マッチ数: 0（閾値を下げて再実行してみてください）")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = TemplateMatchingWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
