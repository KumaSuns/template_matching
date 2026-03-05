# テンプレートマッチング（画像）

画像中でテンプレート画像を検出するサンプルです（OpenCV 使用）。PySide6 の GUI あり。

## GUI の起動

ルートの `python run.py` から「テンプレートマッチング（画像）」を選ぶか、このフォルダで `python gui.py` を実行。

## セットアップ（CLI / GUI 共通）

```bash
pip install -r requirements.txt
```

## 使い方

```bash
python match.py <検索対象画像> <テンプレート画像> [オプション]
```

### オプション

| オプション | 説明 |
|-----------|------|
| `-m`, `--method` | マッチング手法: `TM_CCOEFF_NORMED`（既定）, `TM_CCORR_NORMED`, `TM_SQDIFF_NORMED` |
| `-t`, `--threshold` | マッチとみなす閾値（0〜1、既定: 0.8） |
| `-o`, `--output` | 結果を矩形付きで描画した画像の保存先 |
| `--show` | 結果画像をウィンドウで表示 |

### 例

```bash
# マッチした座標を表示
python match.py scene.png icon.png

# 閾値を下げてマッチ数を増やす
python match.py scene.png icon.png -t 0.7

# 結果を画像で保存して表示
python match.py scene.png icon.png -o result.png --show
```
