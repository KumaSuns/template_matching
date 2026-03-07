"""
フレーム分類のクラス定義。skill の複雑化時は CLASSES を拡張する。
"""
# クラス名のリスト。インデックスがモデル出力の class_id に対応する。
# あとで skill を細かくする場合は "skill_1", "skill_2" などを追加
# none = それ以外。フロー: ready → go → timeup → result
CLASSES = [
    "none",   # それ以外（ready/go/timeup/result 以外）
    "ready",
    "go",
    "timeup",
    "result",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
