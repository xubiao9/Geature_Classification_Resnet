import os
import torch
import random
from PIL import Image
from torchvision import transforms

# 70%当训练集
train_ratio = 0.7
# 剩下的当测试集
test_ratio = 1 - train_ratio
class_all = ["A", "B", "C", "Five", "Point", "V"]
root_data = r"E:/MAX78000FTHR/dataset/Sebastien-Marcel/Marcel-Train/"

train_list, test_list = [], []
data_list = []

for j in range(len(class_all)):
    temp = class_all[j]  # A, ...
    root = root_data + temp + "/"
    for a, b, c in os.walk(root):
        # print(a) # 路径
        # print(b) # 空 []
        # print(c) # 文件夹内所有内容
        for i in range(len(c)):
            data_list.append(os.path.join(a, c[i]))

        for i in range(0, int(len(c) * train_ratio)):
            train_data = os.path.join(a, c[i]) + '\t' + str(j) + '\n'
            train_list.append(train_data)

        for i in range(int(len(c) * train_ratio), len(c)):
            test_data = os.path.join(a, c[i]) + '\t' + str(j) + '\n'
            test_list.append(test_data)

print("len(test):", len(test_list))
print("len(train):", len(train_list))
random.shuffle(train_list)
random.shuffle(test_list)

with open('train_Marcel.txt', 'w', encoding='UTF-8') as f:
    for train_img in train_list:
        f.write(str(train_img))

with open('test_Marcel.txt', 'w', encoding='UTF-8') as f:
    for test_img in test_list:
        f.write(test_img)
