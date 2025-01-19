import cv2
import numpy as np

def ellipse_to_circle(image_path, out_path=None):
    """
    楕円を真円に変換する関数（行列計算を修正）
    """
    image = cv2.imread(image_path)
    if image is None:
        print("画像を読み込めませんでした。")
        return

    # 前処理
    # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # 楕円検出
    center, axes, angle = detect_ellipse(image)

    if center is None:
        print("楕円検出に失敗しました")
        return image, image, (None, None, None)

    # 検出された楕円の可視化
    visualization = draw_detected_shapes(image.copy(), center, axes, angle)

    # 楕円の軸
    major_axis, minor_axis = axes
    target_radius = max(major_axis, minor_axis) / 2  # 半径に修正

    # スケール計算
    scale_x = target_radius / (major_axis / 2)
    scale_y = target_radius / (minor_axis / 2)

    # アフィン変換のための準備
    height, width = image.shape[:2]
    center_x, center_y = center

    # 変換行列の計算（3x3形式）
    angle_rad = np.deg2rad(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # 3x3行列として計算
    # 1. 中心を原点に移動
    T1 = np.array([[1, 0, -center_x],
                   [0, 1, -center_y],
                   [0, 0, 1]], dtype=np.float32)

    # 2. 回転
    R = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta, cos_theta, 0],
                  [0, 0, 1]], dtype=np.float32)

    # 3. スケーリング
    S = np.array([[scale_x, 0, 0],
                  [0, scale_y, 0],
                  [0, 0, 1]], dtype=np.float32)

    # 4. 逆回転
    R_inv = np.array([[cos_theta, sin_theta, 0],
                      [-sin_theta, cos_theta, 0],
                      [0, 0, 1]], dtype=np.float32)

    # 5. 中心を元に戻す
    T2 = np.array([[1, 0, center_x],
                   [0, 1, center_y],
                   [0, 0, 1]], dtype=np.float32)

    # 行列の結合（右から順に適用）
    M = T2 @ R_inv @ S @ R @ T1

    # 2x3行列に変換（アフィン変換用）
    M_affine = M[:2, :]

    # 変換を適用
    transformed = cv2.warpAffine(image, M_affine, (width, height), flags=cv2.INTER_CUBIC)

    if out_path:
        cv2.imwrite(out_path, transformed)

    return transformed


if __name__ == '__main__':
    cd = "../../Resources/"
    input_path = cd + "Input and Output/output/1000-1000.jpg"
    output_path = cd + "Input and Output/output/affine2.jpg"

    ellipse_to_circle(input_path, output_path)
