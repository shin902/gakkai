import numpy as np
import cv2 as cv

# 画像読み込み
float_img = cv.imread('../../Resources/Images/19_57_44/001.jpg')
ref_img = cv.imread('../../Resources/Images/19_57_44/022.jpg')

# ORB検出器作成
orb = cv.ORB_create(nfeatures=10000)

# キーポイントと特徴量の検出
float_kp, float_des = orb.detectAndCompute(float_img, None)
ref_kp, ref_des = orb.detectAndCompute(ref_img, None)

# Brute-Force Matcher
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

# k-NNマッチング
matches = bf.knnMatch(float_des, ref_des, k=2)

# 良好なマッチングを選択
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 十分な対応点があるか確認
if len(good_matches) >= 4:
    # マッチしたキーポイントの抽出
    ref_matched_kpts = np.float32(
        [float_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sensed_matched_kpts = np.float32(
        [ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # ホモグラフィ計算
    H, status = cv.findHomography(
        ref_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0)

    # 画像変換
    warped_image = cv.warpPerspective(
        float_img, H, (float_img.shape[1], float_img.shape[0]))

    cv.imwrite('warped.jpg', warped_image)
else:
    print(f"十分な対応点が見つかりませんでした。マッチ数: {len(good_matches)}")
