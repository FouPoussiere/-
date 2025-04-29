import cv2
import numpy as np
import matplotlib.pyplot as plt
import histogram_equalization as eq
import histogram_matching as mt

if __name__ == "__main__":
    # 1. 自选图像直方图均衡
    eq.histogram_equalization("image1.jpg", "image1")
    eq.histogram_equalization("image2.jpg", "image2")

    # 2. 提供图像素材直方图均衡
    eq.histogram_equalization("image.jpg", "image")

    # 3. 图像直方图匹配
    mt.histogram_matching("image1.jpg", "image2.jpg")
