import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread("image2.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 3)

# 边缘检测
threshold1 = 50
threshold2 = 100
edges = cv2.Canny(gray_image, threshold1, threshold2, apertureSize=3)
edges1 = cv2.Canny(blurred_image, threshold1, threshold2, apertureSize=3)

lines = cv2.HoughLinesP(
    edges1,
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=150,
    maxLineGap=10,
)

# 绘制检测到的直线
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色，线宽为 2


# 显示原图边缘检测和高斯模糊后边缘检测结果
# 显示直线检测结果

plt.figure()
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.figure()
plt.imshow(edges1, cmap="gray")
plt.axis("off")

plt.figure()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
