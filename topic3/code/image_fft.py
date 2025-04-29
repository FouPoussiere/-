import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
imgA = cv2.imread("imageA.jpg", cv2.IMREAD_GRAYSCALE)  # 替换为你的图像 A
imgB = cv2.imread("imageB.jpg", cv2.IMREAD_GRAYSCALE)  # 替换为你的图像 B

# 2. 傅里叶变换
fftA = np.fft.fft2(imgA)
fftB = np.fft.fft2(imgB)

# 中心化频谱
fftA_shifted = np.fft.fftshift(fftA)
fftB_shifted = np.fft.fftshift(fftB)

# 幅度谱和相位谱
magnitude_spectrum_A = np.abs(fftA_shifted)
phase_spectrum_A = np.angle(fftA_shifted)

magnitude_spectrum_B = np.abs(fftB_shifted)
phase_spectrum_B = np.angle(fftB_shifted)

# 显示幅度谱 (对数变换增强显示效果)
plt.figure(figsize=(10, 10))
plt.subplot(221), plt.imshow(np.log(1 + magnitude_spectrum_A), cmap="gray")
plt.title("Magnitude Spectrum A"), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(np.log(1 + magnitude_spectrum_B), cmap="gray")
plt.title("Magnitude Spectrum B"), plt.xticks([]), plt.yticks([])

# 显示相位谱
plt.subplot(223), plt.imshow(phase_spectrum_A, cmap="gray")
plt.title("Phase Spectrum A"), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(phase_spectrum_B, cmap="gray")
plt.title("Phase Spectrum B"), plt.xticks([]), plt.yticks([])
plt.show()


# 3. 逆傅里叶变换 (A的幅度 + B的相位)
combined_spectrum_AB = magnitude_spectrum_A * np.exp(1j * phase_spectrum_B)
combined_spectrum_AB_ishift = np.fft.ifftshift(combined_spectrum_AB)
img_AB_reconstructed = np.fft.ifft2(combined_spectrum_AB_ishift)
img_AB_reconstructed = np.abs(img_AB_reconstructed)  # 取绝对值
img_AB_reconstructed = np.uint8(img_AB_reconstructed)  # 转换为图像格式

# 4. 逆傅里叶变换 (B的幅度 + A的相位)
combined_spectrum_BA = magnitude_spectrum_B * np.exp(1j * phase_spectrum_A)
combined_spectrum_BA_ishift = np.fft.ifftshift(combined_spectrum_BA)
img_BA_reconstructed = np.fft.ifft2(combined_spectrum_BA_ishift)
img_BA_reconstructed = np.abs(img_BA_reconstructed)  # 取绝对值
img_BA_reconstructed = np.uint8(img_BA_reconstructed)  # 转换为图像格式

# 显示重构图像
plt.figure(figsize=(10, 10))
plt.subplot(221), plt.imshow(imgA, cmap="gray")
plt.title("Origin A"), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(imgB, cmap="gray")
plt.title("Origin B"), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_AB_reconstructed, cmap="gray")
plt.title("Reconstructed (A Magnitude, B Phase)"), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_BA_reconstructed, cmap="gray")
plt.title("Reconstructed (B Magnitude, A Phase)"), plt.xticks([]), plt.yticks([])
plt.show()
