"""
テンプレートマッチング: 画像中でテンプレート画像を検出する
"""
import argparse
import cv2
import numpy as np


def match_template(image_path: str, template_path: str, method: str = "TM_CCOEFF_NORMED", threshold: float = 0.8):
    """
    画像内でテンプレートを検索し、一致した位置を返す。

    Args:
        image_path: 検索対象の画像パス
        template_path: テンプレート画像パス
        method: マッチング手法 (TM_CCOEFF_NORMED, TM_CCORR_NORMED, TM_SQDIFF_NORMED など)
        threshold: この値以上をマッチとみなす (0〜1)

    Returns:
        マッチした座標のリスト [(x, y), ...]
    """
    img = cv2.imread(image_path)
    template = cv2.imread(template_path)

    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")
    if template is None:
        raise FileNotFoundError(f"テンプレートを読み込めません: {template_path}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape

    method_map = {
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
    }
    cv_method = method_map.get(method, cv2.TM_CCOEFF_NORMED)

    result = cv2.matchTemplate(img_gray, template_gray, cv_method)

    if cv_method == cv2.TM_SQDIFF_NORMED:
        locations = np.where(result <= 1 - threshold)
    else:
        locations = np.where(result >= threshold)

    points = list(zip(*locations[::-1]))  # (x, y) のリスト
    return points, (w, h), result


def draw_matches(image_path: str, points: list, template_size: tuple, output_path: str | None = None):
    """マッチした位置に矩形を描画する"""
    img = cv2.imread(image_path)
    w, h = template_size
    for pt in points:
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    if output_path:
        cv2.imwrite(output_path, img)
    return img


def main():
    parser = argparse.ArgumentParser(description="テンプレートマッチング")
    parser.add_argument("image", help="検索対象の画像")
    parser.add_argument("template", help="テンプレート画像")
    parser.add_argument("-m", "--method", default="TM_CCOEFF_NORMED",
                        choices=["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"],
                        help="マッチング手法")
    parser.add_argument("-t", "--threshold", type=float, default=0.8, help="マッチ閾値 (0〜1)")
    parser.add_argument("-o", "--output", help="結果を描画した画像の保存先")
    parser.add_argument("--show", action="store_true", help="結果画像を表示")
    args = parser.parse_args()

    points, size, _ = match_template(args.image, args.template, args.method, args.threshold)
    print(f"マッチ数: {len(points)}")
    for i, (x, y) in enumerate(points):
        print(f"  [{i+1}] x={x}, y={y}")

    if points and (args.output or args.show):
        img = draw_matches(args.image, points, size, args.output)
        if args.show:
            cv2.imshow("Template Matching", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
