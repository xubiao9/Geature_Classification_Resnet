import torch
import os, glob
import random, csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm
from torchvision import transforms
from PIL import Image

root = r"E:/MAX78000FTHR/self_proj/train_test_code"

def default_loader(path):
    return Image.open(path).convert('RGB')

class Mydataset_Gesture(Dataset):

    # 参数 root：数据集的位置；resize：修改输入图片的大小；mode：输入图片的模式，train，validation，test
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(Mydataset_Gesture, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        label = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0]))
            label.append((words[1]))
        self.imgs = imgs
        self.label = label
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        fh.close()
        # print("imgs:", len(self.imgs))
        # print("label:", len(self.label))

        # imgs: 3407
        # label: 3407
        # imgs: 1465
        # label: 1465

    def __getitem__(self, idx):
        # idx = idx-1
        # print("idx:", idx)
        fn = self.imgs[idx]
        img = self.loader(fn)
        label = self.label[idx]
        # print("img:", img)  # img: <PIL.Image.Image image mode=RGB size=66x76 at 0x1E3EA418670>
        # print("label:", label)  # label: 1
        # print(type(img))   # PIL.Image.Image
        #
        img = img.resize((64, 64), Image.BILINEAR)
        img = np.asarray(img)

        # img = img.resize((64, 64), Image.ANTIALIAS)
        # img = np.asarray(img).astype('float32')
        # img = img / 255.0

        # label = np.asarray(label)

        # # print(img.shape)
        # img = img.swapaxes(0, 2)
        # img = img.swapaxes(1, 2)
        # print("img.shape:", img.shape) # (64, 64)
        img = Image.fromarray(np.array(img), mode='RGB')
        # label = Image.fromarray(np.array(label))
        label = torch.as_tensor(np.int(label))
        # print(img.shape)

        if self.transform is not None:
            img = self.transform(img)
            # label = self.transform(label)
        # print("img.shape2:", img.shape)  # (3, 64, 64)
        return img, label

    def __len__(self):
        return len(self.imgs)  # 3407

# transform = transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#     ])
#
# train_data = Mydataset_Gesture(txt=root + '\\' + 'train.txt', transform=transforms.ToTensor())
# test_data = Mydataset_Gesture(txt=root + '\\' + 'test.txt', transform=transforms.ToTensor())
#
# #train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
# train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# test_loader = DataLoader(dataset=test_data, batch_size=64)
