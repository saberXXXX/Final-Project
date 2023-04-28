from matplotlib.patches import Rectangle
from torchvision import transforms
from PIL import Image
import os
import time
import torch
import json
import numpy as np
import time
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter
import matplotlib.pyplot as plt
from net.loss import *
from net.network_sn_101 import ACSPNet
from config import Config
from dataloader.loader import *
from util.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate
from sys import exit
import torch.utils.data as data
import pylab

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = Config()
config.test_path = './data/newimage'
config.size_test = (1280, 2560)
config.init_lr = 2e-4
config.offset = True
config.val = True
config.val_frequency = 1
config.teacher = True
config.print_conf()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset
testtransform = Compose([ToTensor(), Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

testdataset = CityPersons(path='./data/citypersons', type='val',
                          config=config, transform=testtransform, preloaded=True)
testloader = DataLoader(testdataset, batch_size=1)
# 原始数据集第一张图片
#data_old = next(iter(testloader))
# 指定位置
for i, data in enumerate(testloader):
    if i == 5:
       data_old = data
       break
print(data_old.shape)


# 自定义数据

# 1. 加载图像
img = Image.open('./data/newimage/1.png')

# 2. 调整图像大小
img = img.resize((2048, 1024))

# 3. 将图像转换为Numpy数组
img_array = np.array(img, dtype=np.float32) / 255.0

# 4. 将图像标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img_array)

# 5. 将图像转换为(batch_size, channels, height, width)格式
data_new = torch.unsqueeze(img_tensor, dim=0).permute(0, 1, 2, 3)

print(data_new.shape)

# 使用原始数据集citypersons中的某张图片
#used_img = data_old
# 自定义图片
used_img = data_new

# 可视化图片
# img_np = used_img.squeeze().permute(1, 2, 0).numpy()  # 将PyTorch张量转换为Numpy数组
# img_np = (img_np * 255).astype(np.uint8)  # 将像素值缩放回0到255的范围
# plt.imshow(img_np)  # 显示图像
# plt.axis('off')  # 关闭坐标轴
# plt.show()  # 显示图像


# net
print('Net...')
net = ACSPNet().to(device)


# position
center = cls_pos().to(device)
height = reg_pos().to(device)
offset = offset_pos().to(device)

teacher_dict = net.state_dict()


def val_single(r, name):

    # 加载模型
    net.eval()
    # load the model here!!!
    teacher_dict = torch.load(name, map_location='cpu')
    net.load_state_dict(teacher_dict)

    print(net)
    print('Perform validation...')
    res = []
    
    # Load single data
    data = used_img

    # 模型前向传播 得到位置、高度和偏移量等参数
    inputs = data.to(device)
    with torch.no_grad():
        pos, height, offset = net(inputs)

    boxes = parse_det_offset(r, pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test, score=0.1, down=4, nms_thresh=0.5)
    if len(boxes) > 0:
        boxes[:, [2, 3]] -= boxes[:, [0, 1]]

        # 反转标准化
        mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
        data = data.cpu() * std + mean

        # 将张量从(batch_size, channels, height, width)格式转换为(height, width, channels)格式
        data = data.squeeze(0).permute(1, 2, 0)

        # 可视化图像
        plt.imshow(data)
        ax = plt.gca()

        for box in boxes:
            temp = dict()
            temp['image_id'] = 1
            temp['category_id'] = 1
            temp['bbox'] = box[:4].tolist()
            temp['score'] = float(box[4])
            res.append(temp)
            x, y, width, height = temp['bbox']
            rect = Rectangle((x, y), width, height, linewidth=2,
                             edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()


    with open('./single_temp_val.json', 'w') as f:
        json.dump(res, f)

    # MRs = validate('./eval_city/val_gt.json', './_temp_val.json')

    # print(name)
    # print('Summarize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
    #       % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))

    # return MRs[0]


name_1 = './models/ACSP(Smooth L1).pth.tea'
name_2 = './models/ACSP(Vanilla L1).pth.tea'
# or Val your own model
#name = './ckpt/ACSP_XXX.pth.tea'
val_single(0.40, name_1)


