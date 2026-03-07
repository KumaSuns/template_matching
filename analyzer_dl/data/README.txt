フレーム分類の学習用データをここに置きます。

train/ と val/ の下に、クラス名のフォルダを作り、その中に画像を入れます。

  train/none/   ← それ以外のフレーム画像（.png, .jpg）
  train/ready/  ← ready のフレーム画像
  train/go/     ← go のフレーム画像
  train/timeup/
  train/result/
  val/none/
  val/ready/
  val/go/
  ...

classes.py の CLASSES に合わせてフォルダ名を用意してください。
