# template_matching

やりたいことをフォルダごとに分けているリポジトリです。各システムは PySide6 で GUI 化されています。

## 起動方法

**ランチャーから起動（推奨）**

```bash
pip install -r requirements.txt   # ルートで PySide6 をインストール
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
