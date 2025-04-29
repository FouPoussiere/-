import cv2

# 读取图像
image = cv2.imread("image.jpg")
# 显示原图像
cv2.imshow("Original Image", image)
# 进行比例缩放
scale_percent = 80  # 比例缩放百分比
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# 缩放图像
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# 显示缩放后的图像
cv2.imwrite("../pics/resized_image.jpg", resized_image)  # 保存图像文件
cv2.imshow("Resized Image", resized_image)
# 裁剪图像（裁剪为左上角的2000x2000区域）
cropped_image = image[0:2000, 0:2000]
cv2.imwrite("../pics/cropped_image.jpg", cropped_image)  # 保存图像文件
cv2.imshow("Cropped Image", cropped_image)
# 翻转图像（水平翻转）
flipped_image = cv2.flip(image, 1)
cv2.imwrite("../pics/flipped_image.jpg", flipped_image)  # 保存图像文件
cv2.imshow("Flipped Image", flipped_image)
# 颜色通道转换（将图像从BGR转换为灰度）
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("../pics/gray_image.jpg", gray_image)  # 保存图像文件
cv2.imshow("Gray Image", gray_image)
# 显示处理后的图像

# 等待按键并关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
