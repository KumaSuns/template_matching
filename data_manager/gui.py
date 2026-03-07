"""
データ管理: analyzer_dl/data 内の画像数表示と動画の削除
"""
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QListWidget,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QFrame,
)

# リポジトリルート（run.py があるディレクトリ）
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "analyzer_dl" / "data"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm")


def count_images_in_dir(path: Path) -> int:
    if not path.is_dir():
        return 0
    n = 0
    for ext in IMAGE_EXTS:
        n += len(list(path.glob(f"*{ext}")))
    return n


def get_data_image_counts() -> list[tuple[str, int]]:
    """(ラベル, 枚数) のリスト。train/val の各クラスフォルダを走査"""
    result = []
    for sub in ("train", "val"):
        base = DATA_DIR / sub
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir():
                n = count_images_in_dir(child)
                result.append((f"{sub} / {child.name}", n))
    return result


def get_total_count(rows: list[tuple[str, int]]) -> int:
    return sum(r[1] for r in rows)


class DataManagerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("データ管理（画像数・動画削除）")
        self.setMinimumSize(500, 480)
        self._settings = QSettings("data_manager", "main")
        self._last_video_dir = self._settings.value("lastVideoDir", str(ROOT), type=str)
        self.setup_ui()
        self.refresh_counts()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- analyzer_dl/data 画像数 ---
        group_data = QGroupBox("analyzer_dl/data 内の画像数")
        data_layout = QVBoxLayout(group_data)
        self.label_data_path = QLabel(str(DATA_DIR))
        self.label_data_path.setStyleSheet("color: #666; font-size: 11px;")
        self.label_data_path.setWordWrap(True)
        data_layout.addWidget(self.label_data_path)
        self.label_counts = QLabel("（更新ボタンで再表示）")
        self.label_counts.setWordWrap(True)
        data_layout.addWidget(self.label_counts)
        btn_refresh = QPushButton("更新")
        btn_refresh.clicked.connect(self.refresh_counts)
        data_layout.addWidget(btn_refresh)
        layout.addWidget(group_data)

        # --- 動画の削除 ---
        group_video = QGroupBox("動画の削除")
        video_layout = QVBoxLayout(group_video)
        row = QHBoxLayout()
        self.label_video_dir = QLabel("フォルダ未選択")
        self.label_video_dir.setStyleSheet("color: #666;")
        self.label_video_dir.setWordWrap(True)
        row.addWidget(self.label_video_dir, 1)
        btn_folder = QPushButton("フォルダを選択...")
        btn_folder.clicked.connect(self.select_video_folder)
        row.addWidget(btn_folder)
        video_layout.addLayout(row)
        self.list_videos = QListWidget()
        self.list_videos.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        video_layout.addWidget(self.list_videos)
        btn_delete = QPushButton("選択した動画を削除")
        btn_delete.clicked.connect(self.delete_selected_videos)
        video_layout.addWidget(btn_delete)
        layout.addWidget(group_video)

    def refresh_counts(self):
        if not DATA_DIR.is_dir():
            self.label_counts.setText(f"フォルダがありません:\n{DATA_DIR}")
            return
        rows = get_data_image_counts()
        total = get_total_count(rows)
        if not rows:
            self.label_counts.setText("(train/val にクラスフォルダがありません)")
            return
        lines = [f"{label}: {n} 枚" for label, n in rows]
        lines.append(f"--- 合計: {total} 枚 ---")
        self.label_counts.setText("\n".join(lines))

    def select_video_folder(self):
        path = QFileDialog.getExistingDirectory(self, "動画フォルダを選択", self._last_video_dir)
        if not path:
            return
        self._last_video_dir = path
        self._settings.setValue("lastVideoDir", path)
        self.label_video_dir.setText(path)
        self.list_videos.clear()
        p = Path(path)
        for ext in VIDEO_EXTS:
            for f in sorted(p.glob(f"*{ext}")):
                if f.is_file():
                    self.list_videos.addItem(str(f))

    def delete_selected_videos(self):
        items = self.list_videos.selectedItems()
        if not items:
            QMessageBox.information(self, "削除", "削除する動画を選択してください。")
            return
        paths = [Path(item.text()) for item in items]
        for p in paths:
            if not p.is_file():
                QMessageBox.warning(self, "削除", f"ファイルがありません: {p}")
                return
        msg = f"次の {len(paths)} 本の動画を削除しますか？\n\n" + "\n".join(p.name for p in paths[:5])
        if len(paths) > 5:
            msg += f"\n... 他 {len(paths) - 5} 本"
        if QMessageBox.question(
            self, "動画を削除",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        failed = []
        for p in paths:
            try:
                p.unlink()
            except Exception as e:
                failed.append(f"{p.name}: {e}")
        if failed:
            QMessageBox.warning(self, "削除", "一部失敗しました:\n" + "\n".join(failed))
        else:
            QMessageBox.information(self, "削除", f"{len(paths)} 本を削除しました。")
        for item in items:
            self.list_videos.takeItem(self.list_videos.row(item))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = DataManagerWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
