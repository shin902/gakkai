import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_snr(original, noisy):
    """
    SNR（信号対雑音比）を計算する関数
    :param original: ノイズなしの元画像（グレースケール）
    :param noisy: ノイズが加わった画像（グレースケール）
    :return: SNR値（dB）
    """
    # 信号の分散
    signal_var = np.var(original)

    # ノイズの計算（元画像とノイズ画像の差分）
    noise = original - noisy
    noise_var = np.var(noise)

    # SNR 計算 (dB 単位)
    snr = 10 * np.log10(signal_var / noise_var)
    return snr

# 画像を読み込む（グレースケール変換）
original_img = cv2.imread('original.jpg', cv2.IMREAD_GRAYSCALE)
noisy_img = cv2.imread('noisy.jpg', cv2.IMREAD_GRAYSCALE)

# SNRを計算
snr_value = calculate_snr(original_img, noisy_img)

print(f"SNR値: {snr_value:.2f} dB")

# 画像の表示
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original_img, cmap='gray')
ax[0].set_title("元画像")
ax[0].axis("off")

ax[1].imshow(noisy_img, cmap='gray')
ax[1].set_title("ノイズ付き画像")
ax[1].axis("off")

plt.show()
