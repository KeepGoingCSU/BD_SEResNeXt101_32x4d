# coding=utf-8
import logging
import os

import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as torchdata
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torchvision import transforms

from dataset.dataset import collate_fn, dataset
from models.multiscale_resnet import multiscale_resnet
from utils.train_util import train, trainlog
from PIL import Image 

rawdata_root = './BD_SEResNeXt101_32x4d/dataset'  # 指定导入图片所在根路径
all_pd = pd.read_csv("./BD_SEResNeXt101_32x4d/dataset/train.txt", sep=" ",
                     header=None, names=['ImageName', 'label'])
train_pd, val_pd = train_test_split(all_pd, test_size=0.15, stratify=all_pd['label'])
print(val_pd.shape)
'''数据扩增'''
data_transforms = {
    'train': transforms.Compose([  # 组合了几种变换方式
        transforms.RandomRotation(degrees=35, resample=Image.BILINEAR, expand=True),  # 图片旋转一定角度
        transforms.RandomResizedCrop(224, scale=(0.49, 1.0), ratio=(0.08, 1.2)),  # 裁剪成224*224
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        #transforms.RandomGrayscale(0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 设定均值和方差去正则化
    ]),
    'val': transforms.Compose([
        transforms.Resize([224,224]),  # 图片大小修改成224*224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

save_dir = './BD_SEResNeXt101_32x4d/models/SEResNeXt101_32x4d'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = '%s/trainlog.log' % save_dir
trainlog(logfile)
data_set = {}
data_set['train'] = dataset(imgroot=os.path.join(rawdata_root, "train"), anno_pd=train_pd,
                            transforms=data_transforms["train"],
                            )
data_set['val'] = dataset(imgroot=os.path.join(rawdata_root, "train"), anno_pd=val_pd,
                          transforms=data_transforms["val"],
                          )
dataloader = {}
dataloader['train'] = torch.utils.data.DataLoader(data_set['train'], batch_size=8, 
                                                  shuffle=True, num_workers=4, collate_fn=collate_fn)
dataloader['val'] = torch.utils.data.DataLoader(data_set['val'], batch_size=4,
                                                shuffle=True, num_workers=4, collate_fn=collate_fn)
'''model'''
# model =resnet50(pretrained=True)
# model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
# model.fc = torch.nn.Linear(model.fc.in_features,100)
model = multiscale_resnet(num_class=100)
base_lr = 0.001
resume = None
if resume:
    logging.info('resuming finetune from %s' % resume)
    model.load_state_dict(torch.load(resume))
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-5)
#optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0)
#optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
best_acc, best_model_wts = train(model,
                                 epoch_num=150,
                                 start_epoch=0,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 exp_lr_scheduler=exp_lr_scheduler,
                                 data_set=data_set,
                                 data_loader=dataloader,
                                 save_dir=save_dir,
                                 print_inter=50,
                                 val_inter=200,
                                 )



