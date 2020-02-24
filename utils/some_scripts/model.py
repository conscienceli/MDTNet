import torch
import torch.nn as nn
from torchvision.models import densenet121
from collections import OrderedDict
from copy import deepcopy
import torch.nn.functional as F

# encoding block
class encoding_block(nn.Module):
    """
    Convolutional batch norm block with relu activation (main block used in the encoding steps)
    """
    def __init__(self, in_size, out_size, kernel_size=3, padding=0, stride=1, 
                 dilation=1, batch_norm=True, dropout=False):
        super().__init__()

        if batch_norm:

            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [nn.ReflectionPad2d(padding=(kernel_size -1)//2),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      nn.PReLU(),
                      nn.BatchNorm2d(out_size),
                      nn.ReflectionPad2d(padding=(kernel_size - 1)//2),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      nn.PReLU(),
                      nn.BatchNorm2d(out_size),
                      ]

        else:
            layers = [nn.ReflectionPad2d(padding=(kernel_size - 1)//2),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      nn.PReLU(),
                      nn.ReflectionPad2d(padding=(kernel_size - 1)//2),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      nn.PReLU(),]

        if dropout:
            layers.append(nn.Dropout())

        self.encoding_block = nn.Sequential(*layers)

    def forward(self, input):

        output = self.encoding_block(input)

        return output


class UNet(nn.Module):
    """
    Main UNet architecture
    """
    def __init__(self, num_classes=1):
        super().__init__()

        # encoding
        self.conv1 = encoding_block(3, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = encoding_block(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = encoding_block(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # self.conv4 = encoding_block(256, 512)
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # # center
        # self.center = encoding_block(512, 1024)


        #cls
        # self.conv5 = nn.Conv2d(512, 512, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.relu5 = nn.ReLU(inplace=True)
        # self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        path_to_check_point = './saved_models/lnet_encoding.pth.tar'
        checkpoint = torch.load(path_to_check_point)
        self.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(path_to_check_point, checkpoint['epoch']))

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input):

        # encoding
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        # conv4 = self.conv4(maxpool3)
        # maxpool4 = self.maxpool4(conv4)

        # # center
        # center = self.center(maxpool4)

       
        return maxpool3


    def load_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


class XNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.unet = UNet()
        self.densenet_temp = densenet121(pretrained=True)

        self.densenet_first_half = nn.Sequential(OrderedDict([
            ('conv0', deepcopy(self.densenet_temp.features.conv0)),
            ('norm0', deepcopy(self.densenet_temp.features.norm0)),
            ('relu0', deepcopy(self.densenet_temp.features.relu0)),
            ('pool0', deepcopy(self.densenet_temp.features.pool0)),
            ('denseblock1', deepcopy(self.densenet_temp.features.denseblock1)),
            ('transition1', deepcopy(self.densenet_temp.features.transition1)),
            ('denseblock2', deepcopy(self.densenet_temp.features.denseblock2)),
            ('transition2', deepcopy(self.densenet_temp.features.transition2)),
            ('denseblock3', deepcopy(self.densenet_temp.features.denseblock3)),
            ('transition3', deepcopy(self.densenet_temp.features.transition3)),
        ]))

        # for param in self.densenet_first_half.parameters():
        #     param.requires_grad = False

        self.densenet_second_half = nn.Sequential(OrderedDict([
            ('denseblock4', deepcopy(self.densenet_temp.features.denseblock4)),
            ('norm5', deepcopy(self.densenet_temp.features.norm5)),
        ]))


        # self.unet_second_half = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # self.final_conv1 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=2, bias=False)
        # self.final_conv2 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=2, bias=False)
        # self.adjust_dim = nn.Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        del self.densenet_temp

        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # features_unet = self.unet(x)
        features = self.densenet_second_half(self.densenet_first_half(x))
        # print(features_unet.shape)
        # features_unet = self.unet_second_half(features_unet)
        # print(features_unet.shape)
        # features_unet = self.unet_second_half(self.unet(x))
        # features_densenet = self.densenet_first_half(x)
        # features_densenet = self.densenet_second_half(self.densenet_first_half(x))
        # features = self.adjust_dim(torch.cat((features_unet, features_densenet), 1))
        # features = self.densenet_second_half(features)
        # features = torch.cat((features_unet, features_densenet), 1)
        # features = self.final_conv2(self.final_conv1(features))


        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
