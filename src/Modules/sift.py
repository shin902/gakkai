import cv2 as cv
import numpy as np
from hsv import get_mask

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
        """特徴点と記述子を取得"""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des

    def match_kp(self, des1, des2, ratio_thresh=0.75):
        """特徴点マッチング"""
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        return good_matches

    def align_images(self, original_image1, original_image2, image1, image2):
        """画像アライメント"""
        kp1, des1 = self.get_kp(image1)
        kp2, des2 = self.get_kp(image2)

        matches = self.match_kp(des1, des2)

        if len(matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            if H is not None:
                h, w = original_image1.shape[:2]
                warped_image = cv.warpPerspective(original_image2, H, (w, h))
                return warped_image, H
            else:
                raise ValueError("ホモグラフィー行列の計算に失敗しました。")
        else:
            raise ValueError("十分なマッチが見つかりませんでした。")

class ImageDenoiser:
    def __init__(self):
        self.default_params = {
            'gaussian': {'ksize': (5,5), 'sigmaX': 0},
            'median': {'ksize': 5},
            'bilateral': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
            'nlmeans': {'h': 10, 'templateWindowSize': 7, 'searchWindowSize': 21}
        }

    def denoise_image(self, image, method='nlmeans', params=None):
        """画像のノイズ除去"""
        if params is None:
            params = {}

        current_params = self.default_params.get(method, {}).copy()
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
