import numpy as np
import cv2

class Ellipse():
    def __init__(self, center, axis, angle_rad):
        self.center = center
        self.axis = axis  # (短軸, 長軸)の順の長さ
        self.angle = angle_rad
        self.ellipse = (center, axis, angle_rad * 180. / np.pi)

    # ダーツの原点座標 (origin) および原点と中心に関する比 (alpha) を出す
    def seg_line(self, lines):
        elm = len(lines)
        a_ell, b_ell = self.axis
        a = 0.5 * a_ell
        b = 0.5 * b_ell
        for i in range(elm - 1):
            rho0, theta0 = lines[i]
            for j in range(i + 1, elm):
                rho1, theta1 = lines[j]
                if abs(theta1 - theta0) > np.pi / 10:
                    break
            o1 = (  np.sin(theta1) * rho0 - np.sin(theta0) * rho1) / np.sin(theta1 - theta0)
            o2 = (- np.cos(theta1) * rho0 + np.cos(theta0) * rho1) / np.sin(theta1 - theta0)
            alpha = np.sqrt((self.center[0] - o1) * (self.center[0] - o1)
                            + (self.center[1] - o2) * (self.center[1] - o2)) / a

            # 得られた直線が原点を通っているための必要条件
            if alpha < np.sqrt(1 - a * a / (b * b)):
                self.origin = (o1, o2)
                self.alpha = alpha
                return

    # 楕円空間の点 (X, Y) を円空間の座標 (x, y) に変換する
    def ellipse2circle(self, X, Y):
        try:
            a_ell, b_ell = self.axis
            a = 0.5 * a_ell
            b = 0.5 * b_ell
            eps = a / b
            sinphi = eps / np.sqrt(1 - self.alpha * self.alpha)
            cosphi = np.sqrt(1 - sinphi * sinphi)
            h = a * cosphi / self.alpha / sinphi
            r = a / (1 - self.alpha) / sinphi
            c = r * (cosphi / h) * (h * cosphi + a * sinphi)

            X1 = np.cos(self.angle) * (X - self.center[0]) + np.sin(self.angle) * (Y - self.center[1])
            Y1 = - np.sin(self.angle) * (X - self.center[0]) + np.cos(self.angle) * (Y - self.center[1])

            x = (X1 * (h + a * sinphi * cosphi) - a * h * cosphi * cosphi) / (X1 * cosphi + h * sinphi) + c
            y = (Y1 * (h * sinphi + a * cosphi)) / (X1 * cosphi + h * sinphi)
            return (x, y)
        except:
            print("The lack of ellipse data!")

# 面積が範囲内の楕円形を見つける.
# contoursは点集合 - つなげると閉曲線.
def findEllipse(contours, minThresE, maxThresE):
    for cnt in contours:
        try:
            Area = cv2.contourArea(cnt)
            if minThresE < Area < maxThresE:
                ellipse = cv2.fitEllipse(cnt)
                # cv2.ellipse(image_proc_img, ellipse, (0, 255, 0), 2)

                x, y = ellipse[0]
                a_ell, b_ell = ellipse[1]
                angle = ellipse[2]

                center_ellipse = (x, y)

                angle_rad = angle * np.pi / 180

                return Ellipse(center_ellipse, (a_ell, b_ell), angle_rad)
        except:
            print("error")

class BoardPicture():
    kernel1 = np.ones((5, 5), np.float32) / 25.
    kernel2 = np.ones((3, 3), np.uint8)
    def __init__(self, imCalBGR):
        self.bgr = imCalBGR
        self.hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)

        blur = cv2.filter2D(self.hsv, -1, BoardPicture.kernel1)
        h, s, imCal = cv2.split(blur)

        ret, thresh = cv2.threshold(imCal, 50, 255, cv2.THRESH_BINARY_INV)

        thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, BoardPicture.kernel2)
        self.thr = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, BoardPicture.kernel2)
        self.edge = cv2.Canny(thresh, 250, 255)

        # ダーツボードの中の閉曲線を contours として得る
        self.contours, hierarchy = cv2.findContours(self.thr, 1, 2)

def gamma_correction(imCal, gamma):
    table = 255 * (np.arange(256) / 255) ** gamma
    table = np.clip(table, 0, 255).astype(np.uint8)

    return cv2.LUT(imCal, table)
