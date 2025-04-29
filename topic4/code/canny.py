import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = "image1.png"
image = cv2.imread(image_path)

# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊以减少噪声
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 设置阈值
threshold1 = 30
threshold2 = 50
# 使用Canny算子进行边缘检测
edges0 = cv2.Canny(gray_image, threshold1, threshold2)
# 高斯模糊后使用Canny算子进行边缘检测
edges1 = cv2.Canny(blurred_image, threshold1, threshold2)

# 显示原图边缘检测和高斯模糊后边缘检测结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Canny Edges")
plt.imshow(edges0, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Gaussian Edges")
plt.imshow(edges1, cmap="gray")
plt.axis("off")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Canny Edges")
plt.imshow(gray_image, cmap="gray")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Canny Edges")
plt.imshow(blurred_image, cmap="gray")
plt.axis("off")

plt.show()
