import cv2
from PIL import Image
from resnet50 import ResNet50, Bottleneck
from resnet18 import ResNet18
import torch
import numpy as np
from torchvision import transforms
import time

# 打开摄像头
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# Tips:1代表打开外置摄像头，0代表电脑内置摄像头（本人使用的是外接摄像头），外置多个摄像头可依此枚举 0，1，2…

# 设定摄像头参数
# width = 1920
# heigth = 1080
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,heigth)
class_all = ["A", "B", "C", "Five", "Point", "V"]
device = torch.device('cuda')
cap = cv2.VideoCapture(0) #设置摄像头 0是默认的摄像头 如果你有多个摄像头的话呢，可以设置1,2,3....
# width = 360
# heigth = 240
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,heigth)

blocks_num = [3, 4, 6, 3]
# model = ResNet50(Bottleneck, blocks_num).to(device)
model = ResNet18().to(device)
# model.load_state_dict(torch.load('./best_Marcel.pth'), strict=False)
# model.load_state_dict(torch.load('best_Marcel.pth'))
# model.load_state_dict(torch.load('./255_save/best_gesture.pth'))
# model.load_state_dict(torch.load('logs/best_self2.pth'))
model.load_state_dict(torch.load('logs/last_selfgg.pth'))
# model.load_state_dict(torch.load('logs/best_gesture.pth'))
model.eval()   # 至关重要，博客已收藏（深度学习 and mydataset）
print("加载权重成功")

while True:
    # 从摄像头中读取一帧图像
    ret, frame = cap.read()
    # ret：表示读取是否成功的布尔值；
    # frame：读取到的图像帧。
    frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
    # print(frame.shape) # (480, 640, 3)
    # 显示图像
    cv2.imshow('IP Camera', frame)
    k = cv2.waitKey(1) & 0xFF

    # 按下q键退出程序
    if k == ord('q'):
        break

    saveFile = r"E:\MAX78000FTHR\self_proj\train_test_code\img_of_camera\test01.jpg"  # 带有中文的保存文件路径
    cv2.imwrite(saveFile, frame)  # imwrite 不支持中文路径和文件名，读取失败，但不会报错!
    print("保存成功")

    # saveFile = r"E:\MAX78000FTHR\self_proj\train_test_code\img_of_camera\test02"  # 带有中文的保存文件路径
    # img_write = cv2.imencode(".jpg", frame)[1].tofile(saveFile)

    save_img = Image.open(saveFile).convert('RGB')
    # save_img = save_img.resize((64, 64), Image.BILINEAR)

    save_img = np.asarray(save_img)
    save_img = Image.fromarray(np.array(save_img), mode='RGB')
    # save_img = save_img[np.newaxis, :, :, :]

    with torch.no_grad():
        data_transfrom = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        save_img = data_transfrom(save_img)
        save_img = save_img.unsqueeze(0).to(device)
        print(save_img.shape)
        logits = model(save_img)
        print("logits:", logits)
        pred = logits.argmax(dim=1)
        print("pred:", pred)
    time.sleep(0.25)




# 释放资源
cap.release()
cv2.destroyAllWindows()# 释放资源

