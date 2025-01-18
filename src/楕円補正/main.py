import board_analysis as ba     # 自作のモジュール
import cv2
import numpy as np

src_path = "./oval.jpg"

src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

# ② 画像の中心座標を求める
height,width  = src_img.shape[:2]
gy = height / 2
gx = width / 2
# print("画像の中心：y={0},x={1}\n".format(gy, gx))

# ③ オブジェクトの重心を求める
object_g = np.array(np.where(src_img == 255)).mean(axis=1)
# print("オブジェクトの中心座標：y={0}, x={1}\n".format(object_g[0], object_g[1]))

# ④ 重心のズレを補正する
dy = gy - object_g[0]
dx = gx - object_g[1]
print("中心座標とのズレ：y={0}, x={1}\n".format(dy, dx))

mat_g = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
affine_img_g = cv2.warpAffine(src_img, mat_g, (width,height))
cv2.imwrite("affine_img_g.jpg", affine_img_g)
