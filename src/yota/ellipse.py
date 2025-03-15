import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def detect_ellipse(image):
    """
    画像から楕円を検出する関数（木星画像用に最適化）
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # コントラスト強調
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
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(largest_contour)
                return ellipse
            except Exception as e:
                print(f"楕円フィッティングエラー: {str(e)}")
    return None

def ellipse_to_circle(image_path, out_path=None):
    """
    楕円を真円に変換し、水平に回転する関数
    """
    image = cv2.imread(image_path)
    if image is None:
        print("画像を読み込めませんでした。")
        return

    ellipse = detect_ellipse(image)
    if ellipse is None:
        print("楕円検出に失敗しました")
        return image

    (center_x, center_y), (major_axis, minor_axis), angle = ellipse
    target_radius = max(major_axis, minor_axis) / 2
    scale_x = target_radius / (major_axis / 2)
    scale_y = target_radius / (minor_axis / 2)
    height, width = image.shape[:2]

    # 変換行列の計算
    M_scale = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    M_scale[0, 0] *= scale_x
    M_scale[0, 1] *= scale_x
    M_scale[1, 0] *= scale_y
    M_scale[1, 1] *= scale_y
    transformed = cv2.warpAffine(image, M_scale, (width, height), flags=cv2.INTER_CUBIC)

    # 水平回転処理
    M_rotate = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1)
    transformed = cv2.warpAffine(transformed, M_rotate, (width, height), flags=cv2.INTER_CUBIC)

    if out_path:
        cv2.imwrite(out_path, transformed)

    return transformed

if __name__ == '__main__':
    # input_path = "./0824.bmp"
    # output_path = "aaaa.jpg"
    # ellipse_to_circle(input_path, output_path)


    input_dir = "../../Resources/Images/19_57_44/"
    output_dir = "../../Resources/Input and Output/affine3"
    os.makedirs(output_dir, exist_ok=True)


    img_paths = sorted(glob(input_dir + "*.jpg"))
    print(img_paths)

    i=0
    for img_path in tqdm(img_paths):
        ellipse_to_circle(img_path, os.path.join(output_dir, str(i)+".jpg"))
        i += 1
