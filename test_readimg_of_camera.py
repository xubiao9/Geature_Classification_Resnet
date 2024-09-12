import cv2
from PIL import Image
import torch

path = "E:/MAX78000FTHR/self_proj/train_test_code/img_of_camera/test01.jpg"
img = Image.open(path).convert('RGB')
# img = img.resize((64, 64))
img = img.resize((64, 64), Image.BILINEAR)

img.show()

