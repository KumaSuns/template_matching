"""
フレーム分類用データセット。
ディレクトリ構成: data/train/<クラス名>/*.png, data/val/<クラス名>/*.png
例: data/train/go/frame_001.png, data/train/none/frame_002.png
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    from torchvision.io import read_image
except ImportError:
    read_image = None

from classes import CLASSES, CLASS_TO_ID


def _load_image_pil(path: Path):
    """PIL で読み torch テンソルに（torchvision がない場合のフォールバック）"""
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def load_image(path: Path) -> torch.Tensor:
    """画像を (3, H, W) float [0,1] で返す"""
    if read_image is not None:
        x = read_image(str(path))
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] == 4:
            x = x[:3]
        return x.float() / 255.0
    return _load_image_pil(path)


class FrameClassDataset(Dataset):
    """data_dir 下に <class_name>/*.png がある構成を想定"""

    def __init__(self, data_dir: str | Path, class_to_id: dict[str, int], transform=None):
        self.data_dir = Path(data_dir)
        self.class_to_id = class_to_id
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for class_name, cid in class_to_id.items():
            folder = self.data_dir / class_name
            if not folder.is_dir():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                for path in sorted(folder.glob(ext)):
                    self.samples.append((path, cid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = load_image(path)
        if self.transform:
            x = self.transform(x)
        return x, label
