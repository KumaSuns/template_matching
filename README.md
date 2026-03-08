# template_matching

やりたいことをフォルダごとに分けているリポジトリです。各システムは PySide6 で GUI 化されています。

## Python バージョン

**PySide6 は Python 3.8〜3.12 を推奨しています。** 3.13 や 3.14 ではビルドが用意されていない場合があります。その場合は Python 3.12 で仮想環境を作ってください（例: `brew install python@3.12` のあと `python3.12 -m venv .venv`）。

## 起動方法

**ランチャーから起動（推奨）**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r analyzer_dl/requirements.txt   # analyzer (DL) を使う場合
python run.py
```

表示されたボタンで、使いたいツールの GUI を開きます。**プロジェクトに `.venv` がある場合、ランチャーは各ツールを .venv の Python で起動します。**（activate せずに `python3 run.py` で起動しても、analyzer (DL) は .venv 内の torch を使います。）

「torch がインストールされていません」と出る場合は、上記のとおり .venv を作成し、`pip install -r requirements.txt` と `pip install -r analyzer_dl/requirements.txt` を実行してください。

**「Numpy is not available」と出る場合**は、OpenCV と NumPy の組み合わせが合っていません。まず **numpy を cv2 より先に読み込む**ようコード側で対応済みです。まだ出る場合は .venv で次を試してください。

1. NumPy と OpenCV を入れ直す:
   ```bash
   .venv/bin/pip install --upgrade "numpy" "opencv-python>=4.10.0"
   ```
2. **「RuntimeError: Numpy is not available」**（検出実行時）の場合は、**バージョン固定**で入れ直す:
   ```bash
   .venv/bin/pip uninstall -y opencv-python opencv-python-headless numpy
   .venv/bin/pip install numpy==1.26.4
   .venv/bin/pip install opencv-python-headless==4.9.0.80
   ```
   または: `sh analyzer_dl/fix_numpy_opencv.sh`

**個別に起動**

```bash
cd analyzer
pip install -r requirements.txt
python gui.py
```

## フォルダ一覧

| フォルダ | 説明 |
|---------|------|
| [analyzer/](analyzer/) | 動画の再生・コマ送り・テンプレート一致検出 |
| [analyzer_dl/](analyzer_dl/) | ディープラーニング版検出用（準備） |
| [video_tool/](video_tool/) | 動画の取り込み・再生・コマ送り・キャプチャ・トリミング |

※ 新しいシステムを追加したら、ルートの `run.py` の `SYSTEMS` にフォルダ名と表示名を追加してください。
