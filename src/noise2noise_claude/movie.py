from noise2noise import Noise2Noise
import torch
import cv2
import os
from tqdm import tqdm
from glob import glob
import numpy as np
from ellipse import ellipse_to_circle


def save_all_frames(video_path, dir_path, basename, ext='jpg'):
	cap = cv2.VideoCapture(video_path)

	if not cap.isOpened():
		return

	os.makedirs(dir_path, exist_ok=True)
	base_path = os.path.join(dir_path, basename)

	digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

	n = 0

	while True:
		ret, frame = cap.read()
		if ret:
			cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
			n += 1
		else:
			return

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


def generate_movie(img_list, img_folder, out_folder, output_path):
	os.makedirs(os.path.join(out_folder, img_folder), exist_ok=True)

	img = cv2.imread(os.path.join(out_folder, img_list[0]))
	h, w = img.shape[:2]
	# 作成する動画
	codec = cv2.VideoWriter_fourcc(*'mp4v')
	#codec = cv2.VideoWriter_fourcc(*'avc1')
	writer = cv2.VideoWriter(output_path, codec, 30000/1001, (w, h),1)

	bar = tqdm(total=frames, dynamic_ncols=True)
	for f in tqdm(img_list):
		path = os.path.join(out_folder, f)
		# 画像を1枚ずつ読み込んで 動画へ出力する
		img = cv2.imread(path)
		writer.write(img)
		bar.update(1)

	bar.close()
	writer.release()


if __name__ == "__main__":
	device = torch.device("mps" if torch.mps.is_available() else "cpu")

	trainer = Noise2Noise(
		train_dir="train_data",
		valid_dir="valid_data",
		model_dir="model_dir",
		device=device
	)
	trainer.load_model('best_model.pth')


	# 各変数の初期設定
	img_folder = "output2"
	out_folder = "output"
	output_path = img_folder + ".mp4"

	# 動画から連番画像を生成する
	# save_all_frames("input.mp4", img_folder)

	img_list = sorted(glob(img_folder+"/*.jpg"))
	print(img_list)
	frames = len(img_list)

	i = 1
	# ノイズ除去画像を生成
	for img_name in tqdm(img_list):
		trainer.denoise_image(img_name, out_folder+str(i)+".jpg")

		img = cv2.imread(os.path.join(out_folder, img_name))
		enhanced = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

		# 楕円を真円に変換
		result, detected, (center, axes, angle) = ellipse_to_circle(enhanced)
		cv2.imwrite("output3/{}".format(img_name), result)

		i += 1




	# 動画を生成
	generate_movie(img_list, img_folder, out_folder, output_path)
