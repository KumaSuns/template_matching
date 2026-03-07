フレーム分類の学習用データをここに置きます。

train/ と val/ の下に、クラス名のフォルダを作り、その中に画像を入れます。

  train/none/   ← 該当なしのフレーム画像（.png, .jpg）
  train/go/     ← go のフレーム画像
  train/timeup/
  train/result/
  val/none/
  val/go/
  ...

classes.py の CLASSES に合わせてフォルダ名を用意してください。
