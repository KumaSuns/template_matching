"""
model.pth を読み込んでエラー内容を表示する。原因調査用。
使い方: cd analyzer_dl && python check_model.py [model.pth のパス]
"""
import sys
from pathlib import Path

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "model.pth"
    path = Path(path)
    if not path.is_file():
        print(f"ファイルがありません: {path}")
        return 1
    try:
        import torch
        from model import build_model
    except ImportError as e:
        print(f"import エラー: {e}")
        return 1
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        print("キー:", list(ckpt.keys()))
        classes = ckpt.get("classes", [])
        num_classes = ckpt.get("num_classes", len(classes))
        print("クラス数:", num_classes, "クラス:", classes)
        state = ckpt.get("model_state_dict")
        if not state:
            print("エラー: model_state_dict がありません")
            return 1
        print("state_dict の先頭キー:", list(state.keys())[:5])
        if next(iter(state.keys()), "").startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model = build_model(num_classes, pretrained=False)
        model.load_state_dict(state, strict=True)
        model.eval()
        print("読み込み成功")
        return 0
    except Exception as e:
        print("エラー:", e)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
