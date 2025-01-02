from matplotlib.image import imsave,imread
import numpy as np
import os,torch
from glob import glob

n = 100 # 作る数
a = 0.3 # ノイズの大きさ
hozon_folder ='noi_gazou' # ノイズを追加された画像を保存するフォルダ
gazou_folder = 'moto_gazou' # 元の画像のフォルダ
gpu = 1 # GPUを使うかどうか

if(not os.path.exists(hozon_folder)):
    os.mkdir(hozon_folder)

for gazou_file in sorted(glob(gazou_folder+'/*')):
    gazou_namae = os.path.basename(gazou_file).split('.')[0]
    gazou = imread(gazou_file) # 画像を読み込み
    if(gpu): # GPUがあればpytorchのテンソルを使う
        gazou = torch.cuda.FloatTensor(gazou)+torch.randn((n,)+gazou.shape).cuda()*255*a
        gazou = torch.clamp(gazou,0,255).type(torch.uint8).cpu().numpy()
    else:
        gazou = gazou+np.random.normal(0,a,(n,)+gazou.shape)*255
        gazou = np.clip(gazou,0,255).astype(np.uint8)

    if(not os.path.exists(os.path.join(hozon_folder,gazou_namae))):
        os.mkdir(os.path.join(hozon_folder,gazou_namae))
    for i,g in enumerate(gazou):
        imsave((os.path.join(hozon_folder,gazou_namae,'%03d.jpg'%i)),g) # 画像を書き込み
    print(gazou_namae)
