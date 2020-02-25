import torch
import torch.nn as nn
from torchvision.models import densenet121
from collections import OrderedDict
from copy import deepcopy
import torch.nn.functional as F

from torchvision.models import resnet50
from torchvision.models import inception_v3
from torchvision.models import densenet121

class EqualLayer(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

    def forward(self, x):
        return x

class BossNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.resnet_model = resnet50(pretrained=True)
        self.resnet_model.fc = nn.Linear(in_features=2048, out_features=4, bias=True)
        checkpoint = torch.load(f'./saved_models/level_cls_resnet_best.pth.tar')
        self.resnet_model.load_state_dict(checkpoint['state_dict'])
        self.resnet_model.fc = EqualLayer()
        for param in self.resnet_model.parameters():
            param.requires_grad = False

        self.inception_model = inception_v3(pretrained=True)
        self.inception_model.fc = nn.Linear(in_features=2048, out_features=4, bias=True)
        checkpoint = torch.load(f'./saved_models/level_cls_inception_best.pth.tar')
        self.inception_model.load_state_dict(checkpoint['state_dict'])
        self.inception_model.fc = EqualLayer()
        for param in self.inception_model.parameters():
            param.requires_grad = False

        self.densenet_model1 = densenet121(pretrained=True)
        self.densenet_model1.classifier = nn.Linear(in_features=1024, out_features=4, bias=True)
        checkpoint = torch.load(f'./saved_models/level_cls_densenet_best.pth.tar')
        self.densenet_model1.load_state_dict(checkpoint['state_dict'])
        self.densenet_model1.classifier = EqualLayer()
        for param in self.densenet_model1.parameters():
            param.requires_grad = False

        # self.focal_model2 = densenet121(pretrained=True)
        # self.focal_model2.classifier = nn.Linear(in_features=1024, out_features=4, bias=True)
        # checkpoint = torch.load(f'./saved_models/level_cls_focal2_best.pth.tar')
        # self.focal_model2.load_state_dict(checkpoint['state_dict'])
        # self.focal_model2.classifier = EqualLayer()
        # for param in self.focal_model2.parameters():
        #     param.requires_grad = False

        # self.focal_model5 = densenet121(pretrained=True)
        # self.focal_model5.classifier = nn.Linear(in_features=1024, out_features=4, bias=True)
        # checkpoint = torch.load(f'./saved_models/level_cls_focal5_best.pth.tar')
        # self.focal_model5.load_state_dict(checkpoint['state_dict'])
        # self.focal_model5.classifier = EqualLayer()
        # for param in self.focal_model5.parameters():
        #     param.requires_grad = False

        self.focal_model1 = densenet121(pretrained=True)
        self.focal_model1.classifier = nn.Linear(in_features=1024, out_features=4, bias=True)
        checkpoint = torch.load(f'./saved_models/level_cls_focal1_best.pth.tar')
        self.focal_model1.load_state_dict(checkpoint['state_dict'])
        self.focal_model1.classifier = EqualLayer()
        for param in self.focal_model1.parameters():
            param.requires_grad = False

        # self.focal_model10 = densenet121(pretrained=True)
        # self.focal_model10.classifier = nn.Linear(in_features=1024, out_features=4, bias=True)
        # checkpoint = torch.load(f'./saved_models/level_cls_focal10_best.pth.tar')
        # self.focal_model10.load_state_dict(checkpoint['state_dict'])
        # self.focal_model10.classifier = EqualLayer()
        # for param in self.focal_model10.parameters():
        #     param.requires_grad = False

        # self.focal_model3 = densenet121(pretrained=True)
        # self.focal_model3.classifier = nn.Linear(in_features=1024, out_features=4, bias=True)
        # checkpoint = torch.load(f'./saved_models/level_cls_focal3_best.pth.tar')
        # self.focal_model3.load_state_dict(checkpoint['state_dict'])
        # self.focal_model3.classifier = EqualLayer()
        # for param in self.focal_model3.parameters():
        #     param.requires_grad = False

        # self.focal_model4 = densenet121(pretrained=True)
        # self.focal_model4.classifier = nn.Linear(in_features=1024, out_features=4, bias=True)
        # checkpoint = torch.load(f'./saved_models/level_cls_focal4_best.pth.tar')
        # self.focal_model4.load_state_dict(checkpoint['state_dict'])
        # self.focal_model4.classifier = EqualLayer()
        # for param in self.focal_model4.parameters():
        #     param.requires_grad = False


        self.boss_layer1 = nn.Linear(2048*2+1024*2, 1024)
        self.boss_layer2 = nn.Linear(1024, 512)

        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        self.inception_model.eval()
        features = torch.cat(
            (
                self.resnet_model(x),
                self.inception_model(x),
                self.densenet_model1(x),
                self.focal_model1(x),
                # self.focal_model2(x),
                # self.focal_model3(x),
                # self.focal_model4(x),
                # self.focal_model5(x),
                # self.focal_model10(x),
            )
            , 1
        )

        out = self.boss_layer2(self.boss_layer1(features))
        out = self.classifier(out)
        
        return out
