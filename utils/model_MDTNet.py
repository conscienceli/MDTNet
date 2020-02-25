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

class MDTNet(nn.Module):
    def __init__(self, num_classes=1, n=0):
        super().__init__()

        self.n = n

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

        self.module_ids = []
        for i in range(1, self.n+1):
            module_id = len(list(self.modules()))
            self.module_ids.append(module_id)
            self.add_module(f'focal_model{i}', densenet121(pretrained=True))
            focal_model = list(self.modules())[module_id]
            focal_model.classifier = nn.Linear(in_features=1024, out_features=4, bias=True)
            checkpoint = torch.load(f'./saved_models/level_cls_focal{i}_best.pth.tar')
            focal_model.load_state_dict(checkpoint['state_dict'])
            focal_model.classifier = EqualLayer()
            for param in focal_model.parameters():
                param.requires_grad = False

        self.fusion_layer1 = nn.Linear(2048*2+1024*(1+n), 1024)
        self.fusion_layer2 = nn.Linear(1024, 512)

        self.classifier = nn.Linear(512, num_classes)


    
    def forward(self, x):
        self.inception_model.eval()
        features = torch.cat(
            (
                self.resnet_model(x),
                self.inception_model(x),
                self.densenet_model1(x),
            )
            , 1
        )
        for i in range(1, self.n+1):
            module_id = len(list(self.modules()))
            focal_model = list(self.modules())[self.module_ids[i-1]]
            features = torch.cat((features, focal_model(x)), 1)

        out = self.fusion_layer2(self.fusion_layer1(features))
        out = self.classifier(out)
        
        return out
