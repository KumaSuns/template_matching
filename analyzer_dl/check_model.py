"""
model.pth の読み込みチェックと、データでの精度チェック。
使い方:
  cd analyzer_dl
  python check_model.py                    # 読み込みのみ
  python check_model.py model.pth          # 読み込みのみ
  python check_model.py model.pth --data_dir ./data   # 読み込み + val で精度表示
"""
import argparse
import sys
from pathlib import Path

def check_load(path: Path):
    """モデル読み込みだけ。成功時は (model, classes), 失敗時は None"""
    if not path.is_file():
        print(f"ファイルがありません: {path}")
        return None
    try:
        import torch
        from model import build_model
    except ImportError as e:
        print(f"import エラー: {e}")
        return None
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        print("キー:", list(ckpt.keys()))
        classes = ckpt.get("classes", [])
        num_classes = ckpt.get("num_classes", len(classes))
        print("クラス数:", num_classes)
        print("クラス一覧(インデックス=予測の番号):", ", ".join(f"{i}:{c}" for i, c in enumerate(classes)))
        try:
            from classes import CLASSES
            if classes != CLASSES:
                print("注意: このモデルのクラス順は、現在の classes.py と異なります。学習時の classes.py の順序で保存されています。")
        except Exception:
            pass
        state = ckpt.get("model_state_dict")
        if not state:
            print("エラー: model_state_dict がありません")
            return None
        if next(iter(state.keys()), "").startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model = build_model(num_classes, pretrained=False)
        model.load_state_dict(state, strict=True)
        model.eval()
        print("読み込み成功")
        return (model, classes)
    except Exception as e:
        print("エラー:", e)
        import traceback
        traceback.print_exc()
        return None


def get_accuracy_result(model, classes: list, data_dir: Path) -> tuple[bool, str]:
    """data_dir/val または data_dir/train で精度を計算し、結果文字列を返す。(成功, メッセージ)"""
    from torch.utils.data import DataLoader
    from train import get_val_transform
    from dataset import FrameClassDataset
    import torch

    data_dir = Path(data_dir)
    val_dir = data_dir / "val"
    train_dir = data_dir / "train"
    if val_dir.is_dir():
        eval_dir = val_dir
        name = "val"
    elif train_dir.is_dir():
        eval_dir = train_dir
        name = "train"
    else:
        return False, f"data/val も data/train もありません:\n{data_dir}"

    class_to_id = {c: i for i, c in enumerate(classes)}
    transform = get_val_transform()
    if transform is None:
        return False, "get_val_transform が使えません"
    try:
        ds = FrameClassDataset(eval_dir, class_to_id, transform=transform)
    except Exception as e:
        return False, f"データセット作成エラー: {e}"
    if len(ds) == 0:
        return False, f"{name} に画像がありません"

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    correct = 0
    total = 0
    per_class_correct = {i: 0 for i in range(len(classes))}
    per_class_total = {i: 0 for i in range(len(classes))}

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for i in range(len(classes)):
                mask = y == i
                per_class_total[i] += mask.sum().item()
                per_class_correct[i] += ((pred == y) & mask).sum().item()

    acc = correct / max(total, 1)
    lines = [f"--- {name} での精度 ---", f"全体: {correct} / {total} = {acc:.2%}"]
    for i, c in enumerate(classes):
        n = per_class_total[i]
        if n > 0:
            a = per_class_correct[i] / n
            lines.append(f"  {c}: {per_class_correct[i]} / {n} = {a:.2%}")
    return True, "\n".join(lines)


def check_accuracy(model, classes: list, data_dir: Path):
    """精度を計算して print（CLI 用）"""
    ok, msg = get_accuracy_result(model, classes, data_dir)
    if ok:
        print("\n" + msg + "\n---")
    else:
        print(msg)


def main():
    parser = argparse.ArgumentParser(description="モデル読み込み・精度チェック")
    parser.add_argument("model", nargs="?", default=None, help="model.pth のパス")
    parser.add_argument("--data_dir", type=str, default=None, help="data/train, data/val があるディレクトリ（指定時は精度を表示）")
    args = parser.parse_args()
    path = Path(args.model) if args.model else Path(__file__).parent / "model.pth"

    result = check_load(path)
    if result is None:
        return 1
    model, classes = result

    if args.data_dir:
        check_accuracy(model, classes, Path(args.data_dir))

    return 0


if __name__ == "__main__":
    sys.exit(main())
