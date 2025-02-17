import cv2 as cv
import numpy as np

class ImageAligner:
    def __init__(self):
        self.sift = cv.SIFT_create()
        self.matcher = cv.BFMatcher()

    def draw_sift_kp(self, image):
        """特徴点を描画"""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kp = self.sift.detect(gray)
        img = cv.drawKeypoints(gray, kp, None, flags=4)
        return img

    def get_kp(self, image):
        """
        特徴点と記述子を取得
        output: kp, des
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.feather = self.sift.detectAndCompute(gray, None)
        return self.feather

    def match_kp(self, des1, des2, ratio_thresh=0.75):
        """特徴点マッチング"""
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        return good_matches

    def draw_matches(self, image1, kp1, image2, kp2, matches):
        """マッチした特徴点を描画"""
        matched_image = cv.drawMatches(image1, kp1, image2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return matched_image

    def align_images(self, original_image1, original_image2, image1, image2):
        """
        2つの画像を位置合わせして変形させる関数

        Args:
            original_image1: 変形の基準となる元画像
            original_image2: 変形させる元画像
            image1: 特徴点検出に使用する画像（マスク済み）
            image2: 特徴点検出に使用する画像（マスク済み）

        Returns:
            tuple: (変形後の画像, ホモグラフィー行列)
        """
        # 特徴点とディスクリプタを取得
        kp1, des1 = self.get_kp(image1)
        kp2, des2 = self.get_kp(image2)

        # 特徴点マッチング
        matches = self.match_kp(des1, des2)

        """マッチした特徴点をファイルに保存
        matched_image = self.draw_matches(original_image1, kp1, original_image2, kp2, matches)
        cv.imwrite('../../Resources/Input and Output/log/matched_image.jpg', matched_image)
        """

        # マッチした特徴点の座標を抽出
        if len(matches) > 4:  # 4点以上のマッチが必要
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # ホモグラフィー行列を計算
            H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

                if H is not None:
                    inlier_count = np.sum(inlier_mask)
                    if inlier_count > len(src_pts) * 0.35:
                        h, w = original_image1.shape[:2]
                        warped_image = cv.warpPerspective(
                            original_image2,
                            H,
                            (w, h),
                            flags=cv.INTER_LINEAR,
                            borderMode=cv.BORDER_REPLICATE
                        )
                        print("変換後のコーナー座標:")
                        # ここで配列を32ビット浮動小数点数に変換
                        transformed_corners = cv.perspectiveTransform(np.array([[[0, 0], [0, h], [w, h], [w, 0]]], dtype=np.float32), H)
                        return warped_image, H
                    else:
                        raise ValueError(f"信頼できるインライアが不足しています: {inlier_count}/{len(src_pts)}")

        raise ValueError(f"アライメントに失敗しました。マッチ数: {len(matches)}")

class ImageDenoiser:
    def __init__(self):
        self.default_params = {
            'gaussian': {'ksize': (5,5), 'sigmaX': 0},
            'median': {'ksize': 5},
            'bilateral': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
            'nlmeans': {'h': 10, 'templateWindowSize': 7, 'searchWindowSize': 21}
        }

    def denoise_image(self, image, params=None):
        """画像のノイズ除去"""
        if params is None:
            params = {}

        current_params = self.default_params.get('nlmeans', {}).copy()
        current_params.update(params)

        denoised = image.copy()
        denoised = cv.fastNlMeansDenoisingColored(
            image,
            None,
            current_params['h'],
            current_params['h'],
            current_params['templateWindowSize'],
            current_params['searchWindowSize']
        )

        return denoised

from hsv import HSVImage

if __name__ == "__main__":
    # 画像の読み込み
    denoiser = ImageDenoiser()

    image1 = cv.imread('../../Resources/Images/19_57_44/001.jpg')
    image2 = cv.imread('../../Resources/Images/19_57_44/002.jpg')
    denoised_image1 = denoiser.denoise_image(image1)
    denoised_image2 = denoiser.denoise_image(image2)




    hsv1 = HSVImage(denoised_image1)
    hsv2 = HSVImage(denoised_image2)
    mask1 = hsv1.get_mask()
    mask2 = hsv2.get_mask()
    masked_image1 = hsv1.make_masked_image()
    masked_image2 = hsv2.make_masked_image()

    aligner = ImageAligner()

    kp1, des1 = aligner.get_kp(masked_image1)
    kp2, des2 = aligner.get_kp(masked_image2)
    print(len(aligner.match_kp(kp1, kp2, des1, des2)))  # kp1, kp2を追加
    # 画像アライメント
    warped_image, _ = aligner.align_images(image1, image2, masked_image1, masked_image2)

    # 画像の保存
    cv.imwrite('../../Resources/Input and Output/log/aligned_denoised.jpg', warped_image)
