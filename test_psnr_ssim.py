import os
import argparse
import re

import numpy as np
import torchvision
import torch.nn as nn
import pytorch_ssim as pytorch_ssim
from PIL import Image
import psnr_ssim
import torch
from torch.autograd import Variable

from utils import make_dataset, edge_compute

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='GCANet')
parser.add_argument('--task', default='dehaze', help='dehaze | derain')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--indir', default='indoor/nyuhaze500/hazy')
parser.add_argument('--outdir', default='output_t')
opt = parser.parse_args()
assert opt.task in ['dehaze', 'derain']

opt.only_residual = opt.task == 'dehaze'

opt.model = 'Improved_GCAnet.pth'

opt.use_cuda = opt.gpu_id >= 0
if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)
test_img_paths = make_dataset(opt.indir)

if opt.network == 'GCANet':
    from improved_GCANet import GCANet

    net = GCANet(in_c=4, out_c=3, only_residual=opt.only_residual)
else:
    print('network structure %s not supported' % opt.network)
    raise ValueError

if opt.use_cuda:
    torch.cuda.set_device(opt.gpu_id)
    net.cuda()
else:
    net.float()

# net.load_state_dict(torch.load(opt.model, map_location='cpu'))
net = nn.DataParallel(net)
net.load_state_dict(torch.load(opt.model)) #加载模型参数，strict设置成False
net.eval()

data_psnr = 0
data_ssim = 0
for img_path in test_img_paths:
    img = Image.open(img_path).convert('RGB')
    im_w, im_h = img.size
    if im_w % 4 != 0 or im_h % 4 != 0:
        img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4)))
    img = np.array(img).astype('float')
    img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
    edge_data = edge_compute(img_data)
    in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128
    in_data = in_data.cuda() if opt.use_cuda else in_data.float()
    with torch.no_grad():
        pred = net(Variable(in_data))
    if opt.only_residual:
        out_img_data = (pred.data[0].cpu().float() + img_data).round().clamp(0, 255)
    else:
        out_img_data = pred.data[0].cpu().float().round().clamp(0, 255)

    out_img = Image.fromarray(out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0))
    # out_img.save(os.path.join(opt.outdir, os.path.splitext(os.path.basename(img_path))[0] + '_%s.png' % opt.task))

     pt = img_path.split('_', 1)[0]
     pt = pt[23:] #截取后面图片名称
     print(pt)
     out_clear = Image.open('./indoor/nyuhaze500/gt/' + pt + '.png').convert('RGB')
     cropimg = torchvision.transforms.CenterCrop((460, 620))
     out_clear = cropimg(out_clear)

     out_clear = np.array(out_clear).astype('float')
     # data_psnr = psnr_ssim.psnr(out_img, out_clear)
     # data_ssim = psnr_ssim.ssim(np.array(out_img).astype('float'), out_clear)
     # print('psnr:' + str(data_psnr))
     # print('ssim:' + str(data_ssim))
     data_psnr += psnr_ssim.psnr(out_img, out_clear)
     data_ssim += psnr_ssim.ssim(np.array(out_img).astype('int'), out_clear)

 print('avg_psnr:' + str(data_psnr / 500))
 print('avg_ssim:' + str(data_ssim / 500))
