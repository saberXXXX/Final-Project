import copy

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

from concate_pic import makevideo
from net.loss import *
from net.network_sn_101 import ACSPNet
from config import Config
from dataloader.loader import *
from util.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate
from sys import exit
import torch.utils.data as data
import pylab
import cv2

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



# 自定义图片
used_img = data_new




# net
print('Net...')
net = ACSPNet().to(device)


# position
center = cls_pos().to(device)
height = reg_pos().to(device)
offset = offset_pos().to(device)

teacher_dict = net.state_dict()


def val_video(r, name):

    # 加载模型
    net.eval()
    # load the model here!!!
    teacher_dict = torch.load(name, map_location='cpu')
    net.load_state_dict(teacher_dict)

    print(net)
    print('Perform validation...')
    res = []
    cap = cv2.VideoCapture("test_.mp4")
    videooutpath = "test_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
    writer = cv2.VideoWriter(videooutpath,fourcc, fps, (2048,1024))
    count = 0
    while cap.isOpened():
        # count += 1
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        img = frame

        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        img = img.resize((2048, 1024))
        # img = np.ascontiguousarray(img)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # writer.write(img)
        # img = cv2.resize(img, (2048, 1024), interpolation=cv2.INTER_CUBIC)
        # 3. 将图像转换为Numpy数组
        img_array = np.array(img, dtype=np.float32) / 255.0

        # 4. 将图像标准化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img_array)

        # 5. 将图像转换为(batch_size, channels, height, width)格式
        data = torch.unsqueeze(img_tensor, dim=0).permute(0, 1, 2, 3)
        # data = data.to(device)
        # with torch.no_grad():

    # # Load single data
    # data = used_img

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

            data = np.rint(data.cpu().numpy()*255)
            # # 可视化图像
            # plt.imshow(data)

            # ax = plt.gca()
            #
            data = np.ascontiguousarray(data)
            for box in boxes:
                temp = dict()
                temp['image_id'] = 1
                temp['category_id'] = 1
                temp['bbox'] = box[:4].tolist()
                temp['score'] = float(box[4])
                res.append(temp)
                x, y, width, height = temp['bbox']
                x_min, x_max = int(x), int(x + width)
                y_min, y_max = int(y), int(y + height)

                cv2.rectangle(data, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=2)
                cv2.putText(data, "person", (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                # rect = Rectangle((x, y), width, height, linewidth=2,
                #                  edgecolor='r', facecolor='none')
                # ax.add_patch(rect)
        writer.write(data)
        cv2.imwrite("test/test_{}.jpg".format(count),data, [cv2.IMWRITE_JPEG_QUALITY,100])
        # plt.savefig("test/plt_test_{}.jpg".format(count))
            # cv2.imshow("video", data)
            #     rect = Rectangle((x, y), width, height, linewidth=2,
            #                      edgecolor='r', facecolor='none')
            #     ax.add_patch(rect)
            # plt.show()


    with open('./single_temp_val.json', 'w') as f:
        json.dump(res, f)
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

name_1 = './models/ACSP(Smooth L1).pth.tea'
name_2 = './models/ACSP(Vanilla L1).pth.tea'

val_video(0.40, name_1)

makevideo()


