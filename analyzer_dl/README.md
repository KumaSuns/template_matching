# analyzer (DL) — フレーム分類

動画フレームを「go / timeup / result / none」などに分類するモデルを学習・推論します。  
あとで skill を細かくする場合は `classes.py` の `CLASSES` にクラスを追加してください。

## 実際の手順（何をすればいいか）

1. **画像を用意する**  
   動画から「go」「timeup」「result」「何でもない」などの場面のフレームを、キャプチャや別ツールで画像として保存する。

2. **フォルダに分けて入れる**  
   `analyzer_dl/data/train/` の下に、クラス名のフォルダ（`none` / `go` / `timeup` / `result`）を作り、その中に該当する画像（.png や .jpg）を入れる。  
   （精度を上げたいときは、同じ構成で `data/val/` にも少しずつ入れておく。）

3. **学習する**  
   - **GUI から**: analyzer (DL) を開き、「学習...」ボタン → データフォルダ・エポック数・保存先を指定 → 「学習開始」。ログと進捗が表示され、終わると「このモデルを読み込みますか？」と出る。
   - **ターミナルから**:
   ```bash
   cd analyzer_dl
   pip install -r requirements.txt
   python train.py --data_dir ./data --epochs 30 --out ./model.pth
   ```
   終わると `model.pth` ができる。

4. **検出で使う**  
   ランチャー（`run.py`）から「analyzer (DL)」を開く → 「モデル: 開く」で `model.pth` を選ぶ → 動画を開く → 「検出実行」を押す。  
   終わったら一覧をクリックして確認し、必要なら「結果を保存（CSV）」で保存する。

**まだデータがない場合**  
まずは `train/none/` と `train/go/` だけ作って、それぞれに画像を数十枚以上入れてから 3 の学習を実行すれば動きます。あとから `timeup` や `result` を増やしてもよい。

## 検出後の流れ

- **一覧の行をクリック** → その場面に飛んで、動画で確認できる
- **「結果を保存（CSV）」** → 検出結果を CSV ファイルで保存する。Excel で開いて一覧で見たり、必要なところだけ使ったりできる

## データセットの置き方

```
analyzer_dl/data/
  train/
    none/    ← 該当なしフレームの画像
    go/      ← go フレームの画像
    timeup/
    result/
  val/       ← 省略可
    none/
    go/
    ...
```

- 各フォルダに `.png` / `.jpg` を入れる。
- クラス名は `classes.py` の `CLASSES` と一致させる。

## 学習のやり方

```bash
cd analyzer_dl
pip install -r requirements.txt
python train.py --data_dir ./data --epochs 20 --out ./model.pth
```

- `--data_dir`: `data/train` と `data/val` があるディレクトリ。
- `--epochs`: エポック数（精度を重視するなら 50 以上も可）。
- `--out`: 保存するモデルファイル（.pth）。
- `--no_pretrained`: ImageNet 事前学習を使わない場合に指定。

**精度まわり**
- 学習時は **ImageNet 正規化** と **データ拡張**（左右反転・色ジッター・軽い回転）を使用しています。推論時も同じ正規化をかけます。
- **val を用意すると**、検証精度が最も高かったエポックのモデルを `--out` に保存します。学習率は CosineAnnealingLR で減衰します。
- 昔の「正規化なし」で学習した model.pth は、この仕様に合わせて **再学習** すると精度が安定します。

## クラスを増やす（skill の複雑化）

`classes.py` の `CLASSES` を編集します。

```python
CLASSES = [
    "none",
    "go",
    "timeup",
    "result",
    "skill_1",   # 追加
    "skill_2",   # 追加
]
```

同じ名前のフォルダを `data/train/` と `data/val/` に作り、画像を入れてから再度 `train.py` を実行します。
