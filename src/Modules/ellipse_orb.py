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

    if len(contours) > 0:
        # 最大の輪郭を見つける
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) >= 5:
            try:
                # 楕円フィッティング
                ellipse = cv2.fitEllipse(largest_contour)
                center = ellipse[0]
                axes = ellipse[1]
                angle = ellipse[2]

                return center, axes, angle, binary

            except Exception as e:
                print(f"楕円フィッティングエラー: {str(e)}")
        else:
            print(f"輪郭の点数が不足しています（{len(largest_contour)} < 5点）")

    return None, None, None, None

def detect_matching_keypoints(image1, image2):
    """
    2つの画像間の特徴点マッチングを行う関数
    """
    # ORB特徴点検出器の初期化
    orb = cv2.ORB_create(nfeatures=500)

    # 特徴点と記述子の検出
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 距離でマッチングをソート
    matches = sorted(matches, key=lambda x: x.distance)

    # 上位の良いマッチングを取得（全マッチの50%）
    good_matches = matches[:len(matches)//2]

    # マッチした特徴点の座標を取得
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return src_pts, dst_pts, good_matches, kp1, kp2

def ellipse_to_circle(image_path, out_path=None, debug=False):
    """
    楕円を真円に変換する関数（特徴点マッチングを利用）
    """
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        print("画像を読み込めませんでした。")
        return None, None, (None, None, None)

    # 楕円検出
    center, axes, angle, binary_image = detect_ellipse(image)

    if center is None:
        print("楕円検出に失敗しました")
        return image, image, (None, None, None)

    # 楕円の軸
    major_axis, minor_axis = axes
    target_radius = max(major_axis, minor_axis) / 2

    # スケール計算
    scale_x = target_radius / (major_axis / 2)
    scale_y = target_radius / (minor_axis / 2)

    # 変換パラメータ
    height, width = image.shape[:2]
    center_x, center_y = center

    # 変換行列の計算
    affine_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # スケーリング行列の追加
    scaled_matrix = affine_matrix.copy()
    scaled_matrix[0, 0] *= scale_x
    scaled_matrix[0, 1] *= scale_y
    scaled_matrix[1, 0] *= scale_x
    scaled_matrix[1, 1] *= scale_y

    # アフィン変換の適用
    transformed = cv2.warpAffine(image, scaled_matrix, (width, height),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REFLECT)

    # デバッグ出力
    if debug:
        # マッチング前後の画像比較
        debug_img = np.hstack([image, transformed])
        cv2.imshow('Original vs Transformed', debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 出力画像の保存
    if out_path:
        cv2.imwrite(out_path, transformed)

    return transformed, binary_image, (center, axes, angle)

def visualize_matching(image_path):
    """
    特徴点マッチングの可視化関数
    """
    # 画像読み込み
    image = cv2.imread(image_path)
    transformed, binary_image, _ = ellipse_to_circle(image_path, debug=False)

    if transformed is None:
        return

    # 特徴点マッチング
    src_pts, dst_pts, good_matches, kp1, kp2 = detect_matching_keypoints(image, transformed)

    # マッチング結果の可視化
    matching_img = cv2.drawMatches(image, kp1,
                                   transformed, kp2,
                                   good_matches,
                                   None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 画像表示
    cv2.imshow('Feature Matching', matching_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_path = "../../Resources/Input and Output/input/S__31277060.jpg"
    output_path = "../../Resources/Input and Output/output/S__31277060_d.jpg"

    # 楕円から円への変換
    ellipse_to_circle(input_path, output_path)

    # 特徴点マッチングの可視化
    visualize_matching(input_path)
