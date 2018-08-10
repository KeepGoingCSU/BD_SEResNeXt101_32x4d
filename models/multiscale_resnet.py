from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
from cnn_finetune import make_model


class multiscale_resnet(nn.Module):
    def __init__(self, num_class):
        super(multiscale_resnet, self).__init__()
        self.resnext101_32x4d = make_model('se_resnext101_32x4d', num_classes=100, pretrained=True)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp_1 = nn.UpsamplingBilinear2d(size=(int(input_size * 0.75) + 1, int(input_size * 0.75) + 1))
        self.interp_2 = nn.UpsamplingBilinear2d(size=(int(input_size * 0.85) + 1, int(input_size * 0.85) + 1))
        self.interp_3 = nn.UpsamplingBilinear2d(size=(int(input_size * 0.95) + 1, int(input_size * 0.95) + 1))
        
        x2 = self.interp_1(x)
        x3 = self.interp_2(x)
        x4 = self.interp_3(x)
        
        x = self.resnext101_32x4d(x)
        x2 = self.resnext101_32x4d(x2)
        x3 = self.resnext101_32x4d(x3)
        x4 = self.resnext101_32x4d(x4)

        out = [x, x2, x3, x4]
        return out
