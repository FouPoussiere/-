import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image_path, title=""):
    """
    读取图像，进行直方图均衡化，并显示原图、均衡化后的图像和直方图。

    Args:
        image_path: 图像文件的路径。
        title: 图像的标题
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像
    if img is None:
        print(f"错误：无法读取图像 {image_path}")

    # 直方图均衡化
    equalized_img = cv2.equalizeHist(img)

    # 计算直方图
    hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

    # 显示图像和直方图
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Original {title}")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(equalized_img, cmap='gray')
    plt.title(f"Equalized {title}")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.plot(hist_original)
    plt.title(f"Original Histogram {title}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)
    plt.plot(hist_equalized)
    plt.title(f"Equalized Histogram {title}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

