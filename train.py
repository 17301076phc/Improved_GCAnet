import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms

from GCANet import GCANet

from dataset import dataset
from torchvision.models import vgg16
from PerceptualLoss import LossNetwork as PerLoss
from torchvision.datasets import ImageFolder


# torch.cuda.empty_cache()


def train(train_loader, model, criterion, optimizer, scheduler, epochs):
    print('start training.................')
    for epoch in range(epochs):
        for step, datas in enumerate(train_loader):
            inputs, target = (datas[0]-128).cuda(), datas[1].cuda()
            # inputs = Variable(inputs, requires_grad=True)
            # target = Variable(target, requires_grad=True)
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss1 = criterion[0](target, y_pred)
            loss2 = criterion[1](target, y_pred)
            loss = loss1 + 0.04 * loss2
            # loss = criterion(y_pred, target=target)

            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('epoch: {}, batch: {}, loss: {}, learning rate:{} '.format(epoch + 1, step + 1, loss.data, optimizer.param_groups[0]['lr']))
        scheduler.step()
    torch.save(model.state_dict(), 'improved_GCAnet.pth')
    print('Finish.................................')


batchsize = 2  # 12 在4个GPU上
trainset = dataset('./hazy/hazy', './clear/clear')
# trainset = ImageFolder('./trans', transform=transform)
train_loader = data.dataloader.DataLoader(dataset=trainset, batch_size=batchsize, shuffle=True, num_workers=4)

model = GCANet().cuda()
model = nn.DataParallel(model, device_ids=[0, 1])
# criterion = nn.MSELoss()
# criterion = criterion.cuda()
criterion = []
criterion.append(nn.L1Loss().cuda())
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()
for param in vgg_model.parameters():
    param.requires_grad = False
criterion.append(PerLoss(vgg_model).cuda())

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)  # 每40epoch decay 0.1
epochs = 50  

if __name__ == '__main__':
    train(train_loader, model, criterion, optimizer, scheduler, epochs)
