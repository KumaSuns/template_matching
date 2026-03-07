"""
フレーム分類のクラス定義。skill の複雑化時は CLASSES を拡張する。
"""
# クラス名のリスト。インデックスがモデル出力の class_id に対応する。
# あとで skill を細かくする場合は "skill_1", "skill_2" などを追加
CLASSES = [
    "none",   # 該当なし
    "go",     # go フレーム
    "timeup", # timeup フレーム
    "result", # result フレーム
    # "skill_1", "skill_2", ...  # 必要に応じて追加
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
