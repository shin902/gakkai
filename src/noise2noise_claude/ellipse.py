import cv2
import numpy as np

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

    print(f"検出された輪郭の数: {len(contours)}")

    if len(contours) > 0:
        # デバッグ用コンテナ
        debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # 最大の輪郭を見つける
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        print(f"最大の輪郭の面積: {largest_area}")

        # 最大の輪郭を緑色で描画
        cv2.drawContours(debug_image, [largest_contour], -1, (0, 255, 0), 2)

        # 点数が十分かチェック
        print(f"輪郭の点数: {len(largest_contour)}")

        if len(largest_contour) >= 5:
            try:
                # 楕円フィッティング
                ellipse = cv2.fitEllipse(largest_contour)
                center = ellipse[0]
                axes = ellipse[1]
                angle = ellipse[2]

                # フィッティングされた楕円を赤色で描画
                cv2.ellipse(debug_image,
                           (int(center[0]), int(center[1])),
                           (int(axes[0]/2), int(axes[1]/2)),
                           angle, 0, 360, (0, 0, 255), 2)

                print(f"楕円フィッティング成功 - 中心: {center}, 軸: {axes}, 角度: {angle}")

                # デバッグ画像を表示
                cv2.imshow('Debug Image', debug_image)

                return center, axes, angle

            except Exception as e:
                print(f"楕円フィッティングエラー: {str(e)}")
        else:
            print(f"輪郭の点数が不足しています（{len(largest_contour)} < 5点）")

    # デバッグ画像を表示
    cv2.imshow('Debug Image', debug_image)
    return None, None, None

def ellipse_to_circle(image):
    """
    楕円を真円に変換する関数
    """
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
    angle_corrected = angle+90  # 回転方向を反転
    M_rotation = cv2.getRotationMatrix2D((center_x, center_y), angle_corrected, 1.0)

    # スケールをアフィン行列に組み込む
    M_rotation[0, 0] *= scale_x
    M_rotation[0, 1] *= scale_y
    M_rotation[1, 0] *= scale_x
    M_rotation[1, 1] *= scale_y

    # 変換を適用
    transformed = cv2.warpAffine(image, M_rotation, (width, height), flags=cv2.INTER_CUBIC)

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

def main(img_path, out_path):
    # 画像の読み込み
    image = cv2.imread(img_path)

    if image is None:
        print("画像を読み込めませんでした。")
        return

    # 前処理
    enhanced = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # 楕円を真円に変換
    result, detected, (center, axes, angle) = ellipse_to_circle(enhanced)

    if center is not None:
        # 結果の表示
        cv2.imshow('Original', image)
        cv2.imshow('Detected', detected)
        cv2.imshow('Result', result)

    cv2.imwrite(out_path, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def

if __name__ == '__main__':
    # main("./995.jpg")
