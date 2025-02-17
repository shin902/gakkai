import cv2 as cv
import numpy as np
from hsv import get_mask



def draw_sift_kp(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(gray)
    img = cv.drawKeypoints(gray, kp, None, flags=4)
    return img

def get_kp(image):
    """
    Returns the keypoints and descriptors of an image using SIFT feature detection.

    Args:
        image: The image to detect keypoints in.

    Returns:
        A tuple of (keypoints, descriptors) detected on the image.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 新しいSIFT実装を使用
    sift = cv.SIFT_create()
    # detectAndCompute を使用して特徴点と特徴量記述子の両方を取得
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def match_kp(des1, des2):
    """
    Matches two sets of descriptors using brute-force matching with ratio test.

    Returns:
        A list of good matches between the two sets of descriptors.
    """
    # BFMatcherを作成（k近傍法を使用）
    bf = cv.BFMatcher()
    # k=2で上位2つのマッチを取得
    matches = bf.knnMatch(des1, des2, k=2)

    # Loweのratio testを適用
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # 0.75は閾値で、調整可能
            good_matches.append(m)

    return good_matches

def align_images(original_image1, original_image2, image1, image2):
    """
    2つの画像を位置合わせして変形させる関数

    Args:
        image1: 基準となる画像
        image2: 変形させる画像

    Returns:
        tuple: (変形後の画像, ホモグラフィー行列)
    """
    # 特徴点とディスクリプタを取得
    kp1, des1 = get_kp(image1)
    kp2, des2 = get_kp(image2)

    # 特徴点マッチング
    matches = match_kp(des1, des2)

    # マッチした特徴点の座標を抽出
    if len(matches) > 4:  # 4点以上のマッチが必要
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # ホモグラフィー行列を計算
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if H is not None:
            # 画像を変形
            h, w = original_image1.shape[:2]
            warped_image = cv.warpPerspective(original_image2, H, (w, h))
            return warped_image, H
        else:
            raise ValueError("ホモグラフィー行列の計算に失敗しました。")
    else:
        raise ValueError("十分なマッチが見つかりませんでした。")

    return None, None

def denoise_image(image, method='gaussian', params=None):
    """
    画像のノイズを除去する関数

    Args:
        image: 入力画像（OpenCV形式）
        method: ノイズ除去の手法 ('gaussian', 'median', 'bilateral', 'nlmeans')
        params: ノイズ除去のパラメータ（辞書形式）
               デフォルト値は手法ごとに設定

    Returns:
        ノイズ除去された画像
    """
    if params is None:
        params = {}

    # 各手法のデフォルトパラメータ
    default_params = {
        'gaussian': {'ksize': (5,5), 'sigmaX': 0},
        'median': {'ksize': 5},
        'bilateral': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
        'nlmeans': {'h': 10, 'templateWindowSize': 7, 'searchWindowSize': 21}
    }

    # パラメータの設定（指定がない場合はデフォルト値を使用）
    current_params = default_params.get(method, {})
    current_params.update(params)

    # 画像のコピーを作成
    denoised = image.copy()
            # Non-local Meansフィルタ（より高度なノイズ除去）
    denoised = cv.fastNlMeansDenoisingColored(image,
                                                None,
                                                current_params['h'],
                                                current_params['h'],
                                                current_params['templateWindowSize'],
                                                current_params['searchWindowSize'])

    return denoised
