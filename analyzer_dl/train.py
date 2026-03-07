"""
フレーム分類モデルの学習スクリプト。

使い方:
  cd analyzer_dl
  pip install torch torchvision
  python train.py --data_dir ./data --epochs 20

データ配置:
  data/train/<クラス名>/*.png  例: data/train/go/, data/train/none/
  data/val/<クラス名>/*.png
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torchvision.transforms import (
        Resize,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        ColorJitter,
        RandomRotation,
    )
except ImportError:
    Resize = None
    Normalize = None
    RandomHorizontalFlip = None
    ColorJitter = None
    RandomRotation = None

from classes import CLASSES, CLASS_TO_ID, NUM_CLASSES
from dataset import FrameClassDataset
from model import build_model, IMAGENET_MEAN, IMAGENET_STD


def get_train_transform():
    """学習用: リサイズ + データ拡張 + ImageNet 正規化"""
    if Resize is None or Normalize is None:
        return None
    return Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        RandomRotation(10),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform():
    """検証用: リサイズ + ImageNet 正規化のみ"""
    if Resize is None or Normalize is None:
        return None
    return Compose([
        Resize((224, 224)),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def run_training(
    data_dir: str | Path,
    epochs: int,
    out_path: str | Path,
    batch_size: int = 32,
    lr: float = 1e-3,
    no_pretrained: bool = False,
    progress_callback=None,
):
    """
    学習を実行する。progress_callback(epoch_1based, total, train_loss, val_acc_or_None) を
    各エポック終了時に呼ぶ。callback が True を返したら中断。
    成功時は None、失敗時はエラーメッセージ str を返す。
    """
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.is_dir():
        return f"{train_dir} がありません。data/train/<クラス名>/*.png を配置してください。"

    train_tf = get_train_transform() or get_val_transform()
    if train_tf is None:
        return "torchvision の transforms が使えません。"
    train_ds = FrameClassDataset(train_dir, CLASS_TO_ID, transform=train_tf)
    val_tf = get_val_transform()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = None
    if val_dir.is_dir():
        val_ds = FrameClassDataset(val_dir, CLASS_TO_ID, transform=val_tf or train_tf)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(NUM_CLASSES, pretrained=not no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = -1.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += x.size(0)
        scheduler.step()
        train_loss = total_loss / max(n, 1)
        val_acc = None
        if val_loader:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            val_acc = correct / max(total, 1)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "num_classes": NUM_CLASSES,
                }, out_path)
        if progress_callback and progress_callback(epoch + 1, epochs, train_loss, val_acc):
            return "中断しました"
    if val_loader is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "num_classes": NUM_CLASSES,
        }, out_path)
    return None


def main():
    parser = argparse.ArgumentParser(description="フレーム分類の学習")
    parser.add_argument("--data_dir", type=str, default="./data", help="data/train, data/val があるディレクトリ")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="./model.pth", help="保存するモデルパス")
    parser.add_argument("--no_pretrained", action="store_true", help="ImageNet 事前学習を使わない")
    args = parser.parse_args()

    def log(epoch, total, train_loss, val_acc):
        msg = f"Epoch {epoch}/{total}  train_loss={train_loss:.4f}"
        if val_acc is not None:
            msg += f"  val_acc={val_acc:.4f}"
        print(msg)
        return False

    err = run_training(
        args.data_dir,
        args.epochs,
        args.out,
        batch_size=args.batch_size,
        lr=args.lr,
        no_pretrained=args.no_pretrained,
        progress_callback=log,
    )
    if err:
        print(f"エラー: {err}")
        return
    print(f"モデルを保存しました: {args.out}")


if __name__ == "__main__":
    main()
