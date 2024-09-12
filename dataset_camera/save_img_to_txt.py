import os
import torch
import random
from PIL import Image
from torchvision import transforms

# 70%当训练集
train_ratio = 0.8
# 剩下的当测试集
test_ratio = 1 - train_ratio
# class_all = ["0", "1", "2", "3", "4", "5", "6"]
class_all = [ "0", "1", "2", "3", "4", "5"]
# root_data = "E:/!_AI_self_Proj/Gesture_Classification_Resnet/dataset_camera/self_dataset/"
root_data = "E:/!_AI_self_Proj/Gesture_Classification_Resnet/dataset_camera/gesture/"

train_list, test_list = [], []
data_list = []

for j in range(len(class_all)):
    temp = class_all[j]  # A, ...
    root = root_data + temp + "/"
    # print(root)
    # print(os.walk(root))
    # a = r"E:/MAX78000FTHR/self_proj/train_test_ code/dataset_camera/self_dataset/"
    # print(a)
    for a, b, c in os.walk(root):
        # print('a:', a) # 路径
        # print(b) # 空 []
        # print(c) # 文件夹内所有内容
        for i in range(len(c)):
            data_list.append(os.path.join(a, c[i]))

        for i in range(0, int(len(c) * train_ratio)):
            train_data = os.path.join(a, c[i]) + ' ' + str(j) + '\n'
            train_list.append(train_data)

        for i in range(int(len(c) * train_ratio), len(c)):
            test_data = os.path.join(a, c[i]) + ' ' + str(j) + '\n'
            test_list.append(test_data)

print("len(test):", len(test_list))
print("len(train):", len(train_list))
random.shuffle(train_list)
random.shuffle(test_list)

with open('traingg.txt', 'w', encoding='UTF-8') as f:
    for train_img in train_list:
        f.write(str(train_img))

with open('testgg.txt', 'w', encoding='UTF-8') as f:
    for test_img in test_list:
        f.write(test_img)
