#改进了视频播放过快的问题
import cv2
# 读取视频文件
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)
# 检查视频是否打开成功
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
# 读取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"原视频帧率: {fps} FPS")
wait_time = int(1000 / fps)  # 计算每帧应等待的时间（毫秒）
print(f"等待时间: {wait_time} ms")
# 播放视频qq
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果没有帧可读，则退出循环
    cv2.imshow('Video', frame)
    # 按'q'键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
