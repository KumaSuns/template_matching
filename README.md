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

表示されたボタンで、使いたいツールの GUI を開きます。

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
