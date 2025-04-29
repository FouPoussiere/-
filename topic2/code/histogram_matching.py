import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram_matching(source, target):
    """进行直方图匹配"""
    source_image = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(target, cv2.IMREAD_GRAYSCALE)

    if source_image is None or target_image is  None:
        print(f"错误：无法读取图像")

    # 计算源图像和目标图像的直方图
    source_hist = cv2.calcHist([source_image], [0], None, [256], [0, 256])
    target_hist = cv2.calcHist([target_image], [0], None, [256], [0, 256])

    # 计算累积分布函数 (CDF)
    source_cdf = np.cumsum(source_hist) / float(source_image.size)
    target_cdf = np.cumsum(target_hist) / float(target_image.size)

    # 创建映射表
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(source_cdf[i] - target_cdf)
        mapping[i] = np.argmin(diff)

    # 应用映射表
    matched_image = mapping[source_image]

    # 显示原始图像、目标图像和匹配后的图像
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(source_image, cmap='gray')
    plt.title('Source Image')
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    hist = cv2.calcHist([source_image], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlim([0, 256])

    plt.subplot(2, 3, 2)
    plt.imshow(target_image, cmap='gray')
    plt.title('Target Image')
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    hist = cv2.calcHist([target_image], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlim([0, 256])

    plt.subplot(2, 3, 3)
    plt.imshow(matched_image, cmap='gray')
    plt.title('Matched Image')
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    hist = cv2.calcHist([matched_image], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()
