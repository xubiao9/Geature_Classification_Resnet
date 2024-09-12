import cv2
from PIL import Image
from resnet50 import ResNet50, Bottleneck
from resnet18 import ResNet18
import torch
import numpy as np
from torchvision import transforms
import time
from torch.utils.data import DataLoader
from mydataset import Mydataset_Gesture

device = torch.device('cuda')
def evalute(model, loader):
    model.eval()

    correct = 0
    total_num = 0
    total = len(loader.dataset)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # print("y:", y)

            logits = model(x)
            # print("logit.shape:", logits.shape)
            # print("logit:", logits)
            pred = logits.argmax(dim=1)
            # if y == "0":
            #     print("logit:", logits)
            # print("pred:", pred)
            # new_label = class_all[pred]
            correct += torch.eq(pred, y).sum().float().item()
            total_num += x.size(0)
    acc = correct / total_num
    return acc


if __name__ == '__main__':
    batchsz = 2
    root = r"E:/MAX78000FTHR/self_proj/train_test_code"

    data_transfrom = {
        "train": transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((64, 64)),
            # transforms.RandomRotation((0, 90), expand=True),
            # torchvision.transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),

            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.ToTensor()
        ])
    }

    train_data = Mydataset_Gesture(txt='E:/MAX78000FTHR/self_proj/train_test_code/dataset_camera' + '\\' + 'train.txt',
                                   transform=data_transfrom["train"])
    test_data = Mydataset_Gesture(txt='E:/MAX78000FTHR/self_proj/train_test_code/dataset_camera' + '\\' + 'test.txt',
                                  transform=data_transfrom["test"])

    # train_data = Mydataset_Gesture(txt=root + '\\' + 'train_Marcel.txt', transform=data_transfrom["train"])
    # test_data = Mydataset_Gesture(txt=root + '\\' + 'test_Marcel.txt', transform=data_transfrom["test"])

    # train_data = Mydataset_Gesture(txt=root + '\\' + 'train.txt', transform=data_transfrom["train"])
    # test_data = Mydataset_Gesture(txt=root + '\\' + 'test.txt', transform=data_transfrom["test"])

    # train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=batchsz, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_data, batch_size=batchsz, num_workers=1)

    blocks_num = [3, 4, 6, 3]
    model = ResNet50(Bottleneck, blocks_num).to(device)
    # model = ResNet18().to(device)
    # model.load_state_dict(torch.load('./best_Marcel.pth'), strict=False)
    # model.load_state_dict(torch.load('E:/MAX78000FTHR/self_proj/train_test_code/best_Marcel.pth'))
    # model.load_state_dict(torch.load('E:/MAX78000FTHR/self_proj/train_test_code/best_gesture.pth'))
    model.load_state_dict(torch.load('E:/MAX78000FTHR/self_proj/train_test_code/best_self.pth'))

    model.eval()  # 至关重要，博客已收藏（深度学习 and mydataset）
    print("加载权重成功")

    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)
    #your code
