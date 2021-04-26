import re

import numpy as np
import torch.utils.data
import torch
import os

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from utils import edge_compute

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    #transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
])


class dataset(torch.utils.data.Dataset):
    def __init__(self, root1, root2):
        imgs = os.listdir(root1)
        clear_imgs = os.listdir(root2)
        clear_imgs.sort(key=lambda x: int(x[:-4]))
        self.imgs = [os.path.join(root1, k) for k in imgs]
        self.clear_imgs = [os.path.join(root2, i) for i in clear_imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pt = img_path.split('_', 1)[0]
        p = re.findall('\d+', pt)[0]
        # print(p)
        clear_img_path = self.clear_imgs[int(p) - 1]
        # print('hazypath:'+img_path+'clearpath:'+clear_img_path)
        img = Image.open(img_path).convert('RGB')
        clear_img = Image.open(clear_img_path).convert('RGB')

        im_w, im_h = img.size
        if im_w % 4 != 0 or im_h % 4 != 0:
            img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4)))
        img = np.array(img).astype('float')
        img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
        edge_data = edge_compute(img_data)
        in_data = torch.cat((img_data, edge_data), dim=0)
        # print(in_data.shape)

        cim_w, cim_h = clear_img.size
        if cim_w % 4 != 0 or cim_h % 4 != 0:
            clear_img = clear_img.resize((int(cim_w // 4 * 4), int(cim_h // 4 * 4)))
        clear_img = np.array(clear_img).astype('float')
        clear_img_data = torch.from_numpy(clear_img.transpose((2, 0, 1))).float()
        # if self.transforms:
        #     in_data = self.transforms(in_data)
        return in_data, clear_img_data - img_data

    def __len__(self):
        return len(self.imgs)
