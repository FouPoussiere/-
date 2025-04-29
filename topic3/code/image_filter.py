import cv2
import numpy as np
import matplotlib.pyplot as plt

# 滤波操作
img = cv2.imread("imageD.jpg", cv2.IMREAD_GRAYSCALE)  # 替换为需要滤波的图像

# 频域滤波
fft = np.fft.fft2(img)
fft_shifted = np.fft.fftshift(fft)


# 创建高通滤波器
def create_highpass_filter(shape, radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - radius : crow + radius, ccol - radius : ccol + radius] = 1
    mask = 1 - mask  # 反转，得到高通
    return mask


# 创建低通滤波器
def create_lowpass_filter(shape, radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - radius : crow + radius, ccol - radius : ccol + radius] = 1
    return mask


# 应用高通滤波器
radius = 30  # 调整半径以控制滤波强度
highpass_mask = create_highpass_filter(img.shape, radius)
fft_filtered_high = fft_shifted * highpass_mask
img_filtered_high = np.fft.ifft2(np.fft.ifftshift(fft_filtered_high))
img_filtered_high = np.abs(img_filtered_high)
img_filtered_high = np.uint8(img_filtered_high)

# 应用低通滤波器
radius = 30  # 调整半径以控制滤波强度
lowpass_mask = create_lowpass_filter(img.shape, radius)
fft_filtered_low = fft_shifted * lowpass_mask
img_filtered_low = np.fft.ifft2(np.fft.ifftshift(fft_filtered_low))
img_filtered_low = np.abs(img_filtered_low)
img_filtered_low = np.uint8(img_filtered_low)

# 空域滤波 (使用 OpenCV 的 filter2D 函数)
# 创建高斯模糊核
kernel_size = (5, 5)  # 调整核大小以控制模糊程度
sigma = 0
blurred_img = cv2.GaussianBlur(img, kernel_size, sigma)

# 创建锐化核
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_img = cv2.filter2D(img, -1, sharpen_kernel)

# 显示滤波结果
plt.figure(figsize=(8, 10))
plt.subplot(321), plt.imshow(img, cmap="gray"), plt.title("Original Image")
plt.subplot(323), plt.imshow(img_filtered_high, cmap="gray"), plt.title(
    "Highpass Filter (Frequency Domain)"
)
plt.subplot(324), plt.imshow(img_filtered_low, cmap="gray"), plt.title(
    "Lowpass Filter (Frequency Domain)"
)
plt.subplot(325), plt.imshow(blurred_img, cmap="gray"), plt.title(
    "Gaussian Blur (Spatial Domain)"
)
plt.subplot(326), plt.imshow(sharpened_img, cmap="gray"), plt.title(
    "Sharpening (Spatial Domain)"
)
plt.show()
