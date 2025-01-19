import cv2
import numpy as np
from glob import glob
import os

def move_center(img_path, out_path):
	src_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

	# ② 画像の中心座標を求める
	height,width  = src_img.shape[:2]
	gy = height / 2
	gx = width / 2
	# print("画像の中心：y={0}, x={1}\n".format(gy, gx))

	# ③ オブジェクトの重心を求める
	object_g = np.array(np.where(src_img == 255)).mean(axis=1)
	# print("オブジェクトの中心座標：y={0}, x={1}\n".format(object_g[0], object_g[1]))

	# ④ 重心のズレを補正する
	dy = gy - object_g[0]
	dx = gx - object_g[1]
	print("中心座標とのズレ: y={0}, x={1}\n".format(dy, dx))

	mat_g = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
	affine_img_g = cv2.warpAffine(src_img, mat_g, (width, height))
	cv2.imwrite(out_path, affine_img_g)


def detect_ellipse(image):
    """
    画像から楕円を検出する関数（木星画像用に最適化）
    """
    # 画像の前処理
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # コントラストの強調
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # ノイズ除去
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    # 2値化
    _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    # モルフォロジー処理
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # デバッグ用コンテナ
        debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # 最大の輪郭を見つける
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) >= 5:
            try:
                # 楕円フィッティング
                ellipse = cv2.fitEllipse(largest_contour)
                center = ellipse[0]
                axes = ellipse[1]
                angle = ellipse[2]

                return center, axes, angle

            except Exception as e:
                print(f"楕円フィッティングエラー: {str(e)}")
        else:
            print(f"輪郭の点数が不足しています（{len(largest_contour)} < 5点）")

    # デバッグ画像を表示
    cv2.imshow('Debug Image', debug_image)
    return None, None, None

def ellipse_to_circle(imag_path, out_path=None):
    """
    楕円を真円に変換する関数
    """
    image = cv2.imread(imag_path)
    if image is None:
        print("画像を読み込めませんでした。")
        return

    # 前処理(enhanced)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


    # 楕円検出
    center, axes, angle = detect_ellipse(image)

    if center is None:
        print("楕円検出に失敗しました")
        return image, image, (None, None, None)

    # 検出された楕円の可視化
    visualization = draw_detected_shapes(image.copy(), center, axes, angle)

    # 楕円の軸
    major_axis, minor_axis = axes
    target_radius = max(major_axis, minor_axis)  # 短軸に合わせる

    # スケール計算
    scale_x = target_radius / major_axis
    scale_y = target_radius / minor_axis

    # アフィン変換行列の作成
    height, width = image.shape[:2]
    center_x, center_y = center

    # 回転行列を生成し、スケールを適用
    angle_corrected = -angle  # 回転方向を反転
    M_rotation = cv2.getRotationMatrix2D((center_x, center_y), angle_corrected, 1.0)

    # スケールをアフィン行列に組み込む
    M_rotation[0, 0] *= scale_x
    M_rotation[0, 1] *= scale_y
    M_rotation[1, 0] *= scale_x
    M_rotation[1, 1] *= scale_y

    # 変換を適用
    transformed = cv2.warpAffine(image, M_rotation, (width, height), flags=cv2.INTER_CUBIC)

    if out_path:
        cv2.imwrite(out_path, transformed)

    return transformed, visualization, (center, axes, angle)

def draw_detected_shapes(image, center, axes, angle, color=(0, 255, 0), thickness=2):
    """
    検出された楕円/円を画像上に描画する関数
    """
    image_with_shape = image.copy()
    if center is not None and axes is not None:
        cv2.ellipse(image_with_shape,
                    (int(center[0]), int(center[1])),
                    (int(axes[0]/2), int(axes[1]/2)),
                    angle,
                    0, 360,
                    color,
                    thickness)

        # 中心点の描画
        cv2.circle(image_with_shape,
                  (int(center[0]), int(center[1])),
                  3,
                  color,
                  -1)

    return image_with_shape



if __name__ == '__main__':
    cd = "../../Resources/"
    input_path = cd + "Input and Output/output/1000-1000.jpg"
    output_path = cd + "Input and Output/output/affinea.bmp"

    ellipse_to_circle(input_path, output_path)
