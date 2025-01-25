from Modules.ellipse import ellipse_to_circle
from Modules.ellipse import generate
from glob import glob
import os
from tqdm import tqdm

if __name__ == '__main__':
    cd = "../../Resources/"
    input_dir = "../../Resources/Images/19_57_44/"
    output_dir = cd + "Input and Output/affine2"
    os.makedirs(output_dir, exist_ok=True)


    img_paths = sorted(glob(input_dir + "*.jpg"))
    print(img_paths)

    i=0
    for img_path in tqdm(img_paths):
        ellipse_to_circle(img_path, os.path.join(output_dir, str(i)+".jpg"))
        i += 1
