"""
ランチャー: 各フォルダのシステム（GUI）を選んで起動する
"""
import os
import subprocess
import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QScrollArea,
    QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# フォルダ名 → 表示名。新しいシステムを追加したらここに追加する
SYSTEMS = [
    ("analyzer", "analyzer"),
    ("analyzer_dl", "analyzer (DL)"),
    ("video_tool", "動画ツール"),
]


def get_root_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_platform_name():
    if sys.platform == "win32":
        return "Windows"
    if sys.platform == "darwin":
        return "Mac"
    return "Linux" if sys.platform.startswith("linux") else "Other"


def get_platform_stylesheet(platform: str) -> str:
    if platform == "Windows":
        return """
            QMainWindow { background-color: #f0f0f0; }
            QLabel#platform { color: #1a1a2e; font-size: 14px; font-weight: bold; }
            QLabel#title { color: #333; }
            QPushButton { background-color: #e0e0e0; border: 1px solid #ccc; border-radius: 4px; }
            QPushButton:hover { background-color: #d0d0d0; }
        """
    if platform == "Mac":
        return """
            QMainWindow { background-color: #f5f5f7; }
            QLabel#platform { color: #1d1d1f; font-size: 14px; font-weight: 600; }
            QLabel#title { color: #424245; }
            QPushButton { background-color: #fff; border: 1px solid #d2d2d7; border-radius: 6px; }
            QPushButton:hover { background-color: #e8e8ed; }
        """
    return ""


def launch_system(folder_name: str):
    root = get_root_dir()
    folder_path = os.path.join(root, folder_name)
    gui_py = os.path.join(folder_path, "gui.py")
    if not os.path.isfile(gui_py):
        return
    subprocess.Popen(
        [sys.executable, "gui.py"],
        cwd=folder_path,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
    )


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ツール一覧")
        self.setMinimumSize(360, 240)
        self._platform = get_platform_name()
        self.setup_ui()
        style = get_platform_stylesheet(self._platform)
        if style:
            self.setStyleSheet(style)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        platform_label = QLabel(self._platform)
        platform_label.setObjectName("platform")
        platform_label.setAlignment(Qt.AlignCenter)
        platform_label.setFont(QFont(None, 14))
        layout.addWidget(platform_label)

        title = QLabel("起動するツールを選んでください")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont(None, 12))
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(8)

        root = get_root_dir()
        for folder_name, display_name in SYSTEMS:
            gui_py = os.path.join(root, folder_name, "gui.py")
            if not os.path.isfile(gui_py):
                continue
            btn = QPushButton(display_name)
            btn.setMinimumHeight(44)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, f=folder_name: launch_system(f))
            scroll_layout.addWidget(btn)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = LauncherWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
