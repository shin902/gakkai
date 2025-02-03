# [toropippi/DigitalImageStability: デジタル画像の「くっきりさ」を測定するやつ](https://github.com/toropippi/DigitalImageStability)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def calculate_color_entropy(image):
    """Calculate Shannon entropy of the color image."""
    flattened = np.reshape(image, (-1, image.shape[-1]))
    unique, counts = np.unique(flattened, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def shuffle_pixels(image):
    """Shuffle pixels in the image randomly."""
    shuffled = image.copy()
    shape = shuffled.shape
    shuffled = shuffled.reshape(-1, shape[-1])
    np.random.shuffle(shuffled)
    return shuffled.reshape(shape)

def flip_pixels(image, num_flips):
    """Randomly flip pixels in the image."""
    flipped = image.copy()
    rows, cols, _ = flipped.shape
    for _ in range(num_flips):
        if np.random.randint(0, 2) == 0:# Randomly decide to flip horizontally or vertically
            r1, c1 = np.random.randint(0, rows - 1), np.random.randint(0, cols)
            flipped[r1, c1], flipped[r1 + 1, c1] = flipped[r1 + 1, c1], flipped[r1, c1]
        else:
            r1, c1 = np.random.randint(0, rows), np.random.randint(0, cols - 1)
            flipped[r1, c1], flipped[r1, c1 + 1] = flipped[r1, c1 + 1], flipped[r1, c1]
    return flipped

def compare_images_ed(img1, img2):
    """
    Compare two images using Euclidean distance.
    Args:
        img1 (numpy.ndarray): The first image (H x W x C).
        img2 (numpy.ndarray): The second image (H x W x C).
    Returns:
        numpy.ndarray: Normalized Euclidean distance per pixel.
    """
    # Ensure the images are of type float for computation
    img1 = img1.astype(np.float64) / np.float64(255.0)
    img2 = img2.astype(np.float64) / np.float64(255.0)
    # Calculate the pixel-wise squared difference
    squared_diff = (img1 - img2) ** 2
    # Sum across color channels (axis=-1) and take the square root
    euclid_dist = np.sqrt(np.sum(squared_diff, axis=-1))
    # Normalize the result by dividing by sqrt(3)
    magnitude = euclid_dist / np.sqrt(3)
    return magnitude


# Parameters
image_name = '../../../Resources/Images/19_57_44/001.jpg' # ここに読み込ませたい画像のファイル名を
num_iter = 10 # 画素数×num_iter回フリップする。この数字を大きくすると計算時間も長くなる
num_avg = 1 # 未実装。ランダムなので結果が毎回かわる。何回の平均を出すか
max_num_flip = 0.1 # 1回のイテレーションでまとめてフリップするピクセルの数/col/raw
nc = 8  # Quantization levels(元ソースでは16だったがpythonだとエラーに。Matlabで実行して大差ないことを確認↓)
# nc = 8 のとき DeltaS_M =0.4274, H = 3.2017
# nc = 16 のとき DeltaS_M = 0.4276, H = 3.2017

# Load image
original_image = Image.open(image_name).convert('RGB')
original_array = np.array(original_image)

# Quantize the image
original_image_quantized = Image.fromarray(original_array).quantize(colors=2**nc)
original_array_quantized = np.array(original_image_quantized.convert('RGB'))

# Calculate entropy
entropy = calculate_color_entropy(original_array_quantized)
print(f"Shannon Entropy: {entropy}")

# Shuffle image
shuffled_image = shuffle_pixels(original_array_quantized)
Image.fromarray(shuffled_image).save(f"{image_name.split('.')[0]}_shuffled.png")

# Initialize results
rows, cols, channels = original_array.shape
norm_factor = rows * cols
num_flips = int(max_num_flip * norm_factor)
num_iter = num_iter * norm_factor

score_original = []
score_shuffled = []

flipped_original = original_array_quantized.copy()
flipped_shuffled = shuffled_image.copy()
for i in range(0, num_iter, num_flips):
    flipped_original = flip_pixels(flipped_original, num_flips)
    flipped_shuffled = flip_pixels(flipped_shuffled, num_flips)

    result_original = compare_images_ed(original_array_quantized, flipped_original)
    result_shuffled = compare_images_ed(shuffled_image, flipped_shuffled)

    one_original = np.sum(result_original) / norm_factor
    score_original.append(one_original)
    one_shuffled = np.sum(result_shuffled) / norm_factor
    score_shuffled.append(one_shuffled)

    print('DeltaS = {0}\t{1}/{2}'.format(one_shuffled - one_original, i, num_iter))

    # 毎iterごとにフリップ画像と色距離マップを保存
    #if i > num_iter - num_flips:
    flipped_img_filename = f"{image_name.split('.')[0]}_flipped_{i}.png"
    Image.fromarray((flipped_original).astype(np.uint8)).save(flipped_img_filename)
    color_distance_map = (result_original * 255).astype(np.uint8)
    color_distance_map_filename = f"{image_name.split('.')[0]}_color_distance_map_{i}.png"
    Image.fromarray(color_distance_map).save(color_distance_map_filename)


# Calculate stability metric
dS = np.array(score_shuffled) - np.array(score_original)

# 論文のΔSMを表示
print("DeltaS_M = {}".format(np.max(dS)))
# Save results
np.savetxt(f"{image_name.split('.')[0]}_stabilityMetric.txt", dS)

# X 軸の値 (反転数 / 総ピクセル数)
x_values = np.arange(0, num_iter, num_flips) / norm_factor
# Stability Metric ΔS のプロット
plt.figure(figsize=(10, 6))
plt.plot(x_values, dS, 'o-', color='#E69F00', label='Stability Metric (ΔS)')
plt.xscale('log')
plt.xlabel('Number of Flips / Total Pixels')
plt.ylabel('Metric Value')
plt.title('Structural Stability Analysis')
plt.legend()
plt.grid()
plt.show()

# 3つのグラフをまとめて表示
plt.figure(figsize=(10, 6))
plt.plot(x_values, score_original, 'o-', color='#0072B2', label='S')
plt.plot(x_values, score_shuffled, 'o-', color='#D55E00', label='S*')
plt.plot(x_values, dS, 'o-', color='#E69F00', label='ΔS')
plt.xscale('log')
plt.xlabel('Number of Flips / Total Pixels')
plt.ylabel('Metric Value')
plt.title('Structural Stability and Scores')
plt.legend()
plt.grid()
plt.show()
