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
    ("video_tool", "動画ツール"),
]


def get_root_dir():
    return os.path.dirname(os.path.abspath(__file__))


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
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        title = QLabel("起動するツールを選んでください")
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
