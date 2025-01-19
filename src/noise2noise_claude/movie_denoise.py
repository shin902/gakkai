from noise2noise import Noise2Noise
import torch
import cv2
import os
from tqdm import tqdm
from glob import glob
import numpy as np

cd = "../../Resources/"


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


def generate_movie(out_folder, mv_path):
	os.makedirs(out_folder, exist_ok=True)

	img_list = sorted(glob(out_folder+"/*.jpg"))

	img = cv2.imread(img_list[0])
	h, w = img.shape[:2]
	# 作成する動画
	codec = cv2.VideoWriter_fourcc(*'mp4v')
	#codec = cv2.VideoWriter_fourcc(*'avc1')
	writer = cv2.VideoWriter(mv_path, codec, 30000/1001, (w, h),1)

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
		train_dir=cd + "AI/train_data",
		valid_dir=cd + "AI/valid_data",
		model_dir=cd + "AI/model_dir",
		device=device
	)
	trainer.load_model('best_model.pth')


	# 各変数の初期設定
	img_folder = cd + "Input and Output/Images"
	out_folder = cd + "Input and Output/output_mv"
	output_path = os.path.join(out_folder, "output.mp4")
	os.makedirs(out_folder, exist_ok=True)

	# 動画から連番画像を生成する
	# save_all_frames("input.mp4", img_folder)

	img_list = sorted(glob(img_folder+"/*.jpg"))
	frames = len(img_list)

	i = 1
	# ノイズ除去画像を生成
	for img_name in tqdm(img_list):
		trainer.denoise_image(img_name, os.path.join(out_folder, str(i)+".jpg"))
		i += 1




	# 動画を生成
	generate_movie(out_folder, output_path)
