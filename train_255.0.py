import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader
from mydataset import Mydataset_Gesture
from torchvision import transforms
from resnet50 import ResNet50, Bottleneck
from resnet18 import ResNet18
import tqdm
import os
# from pokemon import Pokemon

# from    resnet import ResNet18
from torchvision.models import resnet50

# 加载 torchvision.models 和我们自己的区别就是，包里面的是训练的一个状态
# 可以直接得到一个 train 好的一个model
# from utils import Flatten
class_all = ["A", "B", "C", "Five", "Point", "V"]
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
            if y == "0":
                print("logit:", logits)
            # print("pred:", pred)
            # new_label = class_all[pred]
            correct += torch.eq(pred, y).sum().float().item()
            total_num += x.size(0)
    acc = correct / total_num
    return acc


def main():
    root = r"E:/MAX78000FTHR/self_proj/train_test_code"

    batchsz = 2
    lr = 1e-3
    epochs = 20


    print("Using {} device training.".format(device.type))
    torch.manual_seed(1234)

    data_transfrom = {
        "train": transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((64, 64)),
            # transforms.RandomRotation((0, 90), expand=True),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),

            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.ToTensor()
        ])
    }

    # transform = transforms.Compose([
    #                 transforms.Resize((64, 64)),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    #         ])

    # ------------------------------------------------------------
    # 两份数据集训练测试，步骤如下
    # （1）数据载入换 txt 文件
    # （2）换模型，Marcel数据集用 resnet50，土耳其的数据集用 resnet18 (101行)
    # （3）修改权重保存文件的文件名（162行）
    # 注意：transform 不要归一化
    # 注意：resnet50 的 class_num = 6 ， 因为数据集是6个手势
    #      resnet18 的 class_num = 10， 10个手势
    #      根据训练的数据集修改修改
    # ------------------------------------------------------------

    # train_data = Mydataset_Gesture(txt='E:/MAX78000FTHR/self_proj/train_test_code/dataset_camera' + '\\' + 'train.txt', transform=data_transfrom["train"])
    # test_data = Mydataset_Gesture(txt='E:/MAX78000FTHR/self_proj/train_test_code/dataset_camera' + '\\' + 'test.txt', transform=data_transfrom["test"])

    # train_data = Mydataset_Gesture(txt=root + '\\' + 'train_Marcel.txt', transform=data_transfrom["train"])
    # test_data = Mydataset_Gesture(txt=root + '\\' + 'test_Marcel.txt', transform=data_transfrom["test"])

    train_data = Mydataset_Gesture(txt=root + '\\' + 'train.txt', transform=data_transfrom["train"])
    test_data = Mydataset_Gesture(txt=root + '\\' + 'test.txt', transform=data_transfrom["test"])

    # train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=batchsz, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_data, batch_size=batchsz, num_workers=1)
    print(len(train_loader))
    print(len(test_loader))

    print('加载成功！')

    x, label = iter(train_loader).next()
    print('x.shape:', x.shape, 'label.shape:', label.shape)
    blocks_num = [3, 4, 6, 3]
    # model = ResNet50(Bottleneck, blocks_num).to(device)
    model = ResNet18().to(device)
    if os.path.exists("./255_save/best_gesture.pth"):
        print("已加载权重")
        model.load_state_dict(torch.load("./255_save/best_gesture.pth"))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss().to(device)
    print(model)

    best_acc, best_epoch = 0, 0
    global_step = 0

    # viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    # viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    print("开始训练")
    print('------------------------------------')
    for epoch in range(epochs):
        print("epoch:", epoch)
        model.train()
        for step, (x, y) in enumerate(tqdm.tqdm(train_loader, desc="training_batch=2")):
            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1
        print('epoch:', epoch, 'loss:', loss.item())
        model.eval()

        correct = 0
        total_num = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                print("y:", y)

                logits = model(x)
                # print("logit.shape:", logits.shape)
                # print("logit:", logits)
                pred = logits.argmax(dim=1)
                print("pred:", pred)
                # new_label = class_all[pred]
                correct += torch.eq(pred, y).sum().float().item()
                total_num += x.size(0)
            print("\n")
            # print("y:", y)
            # print("logit:", logits)
            # print("y.shape:", y.shape)
            # print("logit.shape:", logits.shape)
            # print("y:", y)
            # print("pred:", pred)
            acc = correct / total_num
            print('epoch:', epoch, 'acc : ', acc)

        if epoch % 1 == 0:
            if acc > best_acc:
                best_epoch = epoch
                best_acc = acc
                # torch.save(model.state_dict(), './255_save/best_self.pth')
                torch.save(model.state_dict(), './255_save/best_gesture.pth')
                # torch.save(model.state_dict(), './255_save/best_Marcel.pth')  # mdl 文件也是权重保存的文件格式

                # if os.path.exists("./255_save/best_self.pth"):
                #     os.remove("./255_save/best_self.pth")
                    # torch.save(model.state_dict(), './255_save/best_self.pth')
                    # torch.save(model.state_dict(), './255_save/best_gesture.pth')
                    # torch.save(model.state_dict(), './255_save/best_Marcel.pth')  # mdl 文件也是权重保存的文件格式

                print("已保存权重")

                # 1.保存整个网络
                # torch.save(model_object, 'model.pth')
                # 1.1加载参数
                # model = torch.load('model.pth')

        print("\n")
        print('best acc:', best_acc, 'best epoch:', best_epoch)
        # print("\n")

        # 本次训练未用到 val，若用到可添加下面代码，载入权重测试
        # model.load_state_dict(torch.load('best.mdl'))
        # print('loaded from ckpt!')
        #
        # test_acc = evalute(model, test_loader)
        # print('test acc:', test_acc)




        # if epoch % 1 == 0:
        #     test_acc = evalute(model, test_loader)
        #     print("current_acc:", test_acc)
        #     if test_acc > best_acc:
        #         best_epoch = epoch
        #         best_acc = test_acc
        #
        #         # torch.save(model.state_dict(), 'best.mdl')
        #
        #         # viz.line([val_acc], [global_step], win='val_acc', update='append')
        #
        # print('best acc:', best_acc, 'best epoch:', best_epoch)

        # model.load_state_dict(torch.load('best.mdl'))
        # print('loaded from ckpt!')
        #
        # test_acc = evalute(model, test_loader)
        # print('test acc:', test_acc)


    #     if epoch % 1 == 0:
    #
    #         val_acc = evalute(model, val_loader)
    #         if val_acc > best_acc:
    #             best_epoch = epoch
    #             best_acc = val_acc
    #
    #             torch.save(model.state_dict(), 'best.mdl')
    #
    #             # viz.line([val_acc], [global_step], win='val_acc', update='append')
    #
    # print('best acc:', best_acc, 'best epoch:', best_epoch)
    #
    # model.load_state_dict(torch.load('best.mdl'))
    # print('loaded from ckpt!')
    #
    # test_acc = evalute(model, test_loader)
    # print('test acc:', test_acc)


if __name__ == '__main__':
    main()
