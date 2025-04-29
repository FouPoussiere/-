import cv2
# 读取视频文件
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)
# 检查视频是否打开成功
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
# 播放视频
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果没有帧可读，则退出循环
    cv2.imshow('Video', frame)
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
