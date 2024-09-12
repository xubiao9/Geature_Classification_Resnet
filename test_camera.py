import cv2

cap = cv2.VideoCapture(0) #设置摄像头 0是默认的摄像头 如果你有多个摄像头的话呢，可以设置1,2,3....

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
    cv2.imshow('IP Camera', frame)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()# 释放资源