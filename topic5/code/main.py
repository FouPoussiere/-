import numpy as np
import cv2
import matplotlib.pyplot as plt


def motion_blur_psf(img_size, length, angle):
    """生成运动模糊的点扩散函数（PSF）"""
    # 创建PSF
    psf = np.zeros(img_size)
    h, w = img_size
    x_center = int((h - 1) / 2)
    y_center = int((w - 1) / 2)  # 图像中心坐标

    # 将angle角度上length个点置成1
    for i in range(length):
        delta_x = round(np.sin(angle) * i)
        delta_y = round(np.cos(angle) * i)
        psf[int(x_center - delta_x), int(y_center + delta_y)] = 1

    psf /= psf.sum()  # 归一化
    return psf


def wiener_filter(blurred, psf, K):
    """应用维纳滤波器进行复原"""
    # 将图像和PSF转换到频域
    blurred_fft = np.fft.fft2(blurred)
    psf_fft = np.fft.fft2(psf)

    # 计算维纳滤波器
    wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + K)

    # 应用滤波器
    restored_fft = blurred_fft * wiener_filter
    restored = np.fft.ifftshift(
        np.fft.ifft2(restored_fft)
    )  # 使像频按中心对称逆傅立叶变换

    return np.abs(restored)


if __name__ == "__main__":
    # 读取模糊图像
    blurred_image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

    # 生成PSF
    length = 30
    angle = 11 / 180 * np.pi
    psf = motion_blur_psf(blurred_image.shape, length, angle)

    # 不同信噪比的复原效果
    K = 0.01  # 信噪比的倒数
    restored_image1 = wiener_filter(blurred_image, psf, K)
    K = 0.02
    restored_image2 = wiener_filter(blurred_image, psf, K)
    K = 0.05
    restored_image5 = wiener_filter(blurred_image, psf, K)

    # 预处理
    gaussian_image = cv2.GaussianBlur(blurred_image, (3, 3), 0)
    median_image = cv2.medianBlur(blurred_image, 3)
    # 应用维纳滤波器进行复原
    K = 0.02  # 信噪比的倒数
    restored_gaussian_image = wiener_filter(gaussian_image, psf, K)
    restored_median_image = wiener_filter(median_image, psf, K)

    # 显示原图像、PSF图像、复原图像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Blurred Image")
    plt.imshow(blurred_image, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("PSF Image")
    plt.imshow(psf, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("Restored Image")
    plt.imshow(restored_image2, cmap="gray")
    plt.axis("off")
    plt.show()

    # 显示不同信噪比的对比结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("K=0.01")
    plt.imshow(restored_image1, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("K=0.02")
    plt.imshow(restored_image2, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("K=0.05")
    plt.imshow(restored_image5, cmap="gray")
    plt.axis("off")
    plt.show()

    # 显示预处理后的复原效果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Origin")
    plt.imshow(restored_image2, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Gaussion")
    plt.imshow(restored_gaussian_image, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("Median")
    plt.imshow(restored_median_image, cmap="gray")
    plt.axis("off")
    plt.show()
