#!/bin/sh
# 「RuntimeError: Numpy is not available」対策
# 使い方: プロジェクトルートで .venv/bin/pip がある状態で
#   ./analyzer_dl/fix_numpy_opencv.sh
# または
#   sh analyzer_dl/fix_numpy_opencv.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIP="${ROOT}/.venv/bin/pip"
if [ ! -x "$PIP" ]; then
  echo "Error: $PIP が見つかりません。.venv を作成してから実行してください。"
  exit 1
fi

echo "opencv と numpy をアンインストールしています..."
"$PIP" uninstall -y opencv-python opencv-python-headless numpy 2>/dev/null || true

echo "numpy 1.26.4 をインストールしています..."
"$PIP" install "numpy==1.26.4"

echo "opencv-python-headless 4.9.0.80 をインストールしています..."
"$PIP" install "opencv-python-headless==4.9.0.80"

echo "完了。.venv/bin/python run.py で起動して検出実行を試してください。"
