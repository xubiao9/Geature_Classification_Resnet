import cv2
from PIL import Image
from resnet50 import ResNet50, Bottleneck
from resnet18 import ResNet18
import torch
import numpy as np
from torchvision import transforms
import time

cap = cv2.VideoCapture(0)

def img_to_64(frame):
    saveFile = r"E:\!_AI_self_Proj\Gesture_Classification_Resnet\img_of_camera\test02.jpg"  # 带有中文的保存文件路径
    cv2.imwrite(saveFile, frame)  # imwrite 不支持中文路径和文件名，读取失败，但不会报错!
    save_img = Image.open(saveFile).convert('RGB')
    save_img = save_img.resize((64, 64), Image.BILINEAR)
    r_image = cv2.cvtColor(np.array(save_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(saveFile, r_image)

    return r_image

while True:
    ret, frame = cap.read()
    # ret：表示读取是否成功的布尔值；
    # frame：读取到的图像帧。
    frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
    img = img_to_64(frame) # 获得转化成64*64的图像


    cv2.imshow('IP Camera', frame)
    k = cv2.waitKey(1) & 0xFF
    # 按下q键退出程序
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()# 释放资源