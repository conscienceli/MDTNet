# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

#%% Configuration
test_image_num = 5

config = 'pre_trained_resnet50'
# config = 'pre_trained_inception_v3'
# config = 'pre_trained_densenet'

import torch
import torch.nn as nn
import os

if config == "pre_trained_resnet50":
    from torchvision.models import resnet50
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    #for gc only
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_name = 'validation_cls_resnet'
    BATCH_SIZE = 96
    NUM_WORKERS = 8

elif config == "pre_trained_inception_v3":
    from torchvision.models import inception_v3
    model = inception_v3(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    #for gc only
    # model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, bias=False, kernel_size=3, stride=2)
    # model.transform_input = False
    model_name = 'validation_cls_inception'
    BATCH_SIZE = 128
    NUM_WORKERS = 8

elif config == "pre_trained_densenet":
    from torchvision.models import densenet121
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(in_features=1024, out_features=1, bias=True)
    #for gc only
    # model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_name = 'validation_cls_densenet'
    BATCH_SIZE = 64
    NUM_WORKERS = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# %% Training 

from utils import train
from utils.dataset_validation import gen_train_loaders
import torch.nn.functional as F

dataloaders_all = gen_train_loaders(BATCH_SIZE, NUM_WORKERS, test_image_num=test_image_num)

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-4)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)
loss_func = F.binary_cross_entropy_with_logits
model = train.train_model(model_name, model, dataloaders_all, device, optimizer_ft, loss_func, exp_lr_scheduler, num_epochs=20, regression=True)


# %% Inference

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from utils.vis_cam import vis_cam
import sklearn.metrics
from tqdm import tqdm
import matplotlib.pyplot as plt

checkpoint = torch.load(f'./saved_models/{model_name}_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()   # Set model to evaluate mode
os.makedirs('./results/', exist_ok=True)

images = []
labels = []

csv_path = './data/patch - patch_list.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    records = f.read()
records_number = len(records.split('\n'))
records = records.split('\n')[-test_image_num:]

for record in records:
    record = record.split(',')
    severity = int(record[4])
    type_label = np.ones((1))
    if severity > 3 or severity < 0:
        type_label[0] = 0
    path = record[3].split('/')[2:]
    path = './data/Archive_cropped/' + '/'.join(path)
    images.append(path)
    labels.append(type_label)

corrects = 0.
tp = 0.
tn = 0.
fp = 0.
fn = 0.
Y_true = []
Y_pred = []
model = model.to(device)
for image_id, image in enumerate(tqdm(images)):
    image = Image.open(image)
    raw_image = image.copy()

    #for gc only
    # image = np.array(image)[:,:,1][...,np.newaxis]

    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    preds = model(image)
    preds = nn.functional.sigmoid(preds)
    # print(preds.item(), labels[image_id])

    Y_true.append(labels[image_id])
    Y_pred.append(preds.item())

    isCorrect = 'Wrong'
    if (preds.item() >= 0.5) == (labels[image_id] > 0):
        corrects += 1.
        isCorrect = 'True'
    
    if preds.item() >= 0.5 and labels[image_id] > 0:
        tp += 1.
    elif preds.item() < 0.5 and labels[image_id] > 0:
        fp += 1.
    elif preds.item() >= 0.5 and labels[image_id] == 0:
        fn += 1.
    elif preds.item() < 0.5 and labels[image_id] == 0:
        tn += 1.

    raw_image.save(f'results/ID({image_id})-PR({preds.item()})-GT({labels[image_id]}).png')

    # model = model.cpu()
    # vis_cam(device, model, raw_image, 0, 7, f'ID({image_id})-PR({preds.item()})-GT({labels[image_id]})')
    # model = model.cuda()

print('\naccuracy: ', corrects/ float(len(images)))
print('precision: ', tp/(tp+fp))
print('recall: ', tp/(tp+fn))
    # break
print('AP: ', sklearn.metrics.average_precision_score(Y_true, Y_pred))


# %%

import tensorly as tl
from thop import profile,clever_format
import time
from tqdm import tqdm

model.cpu()
model.eval()
tl.set_backend('pytorch')

inputs = torch.randn(1, 3, 345, 345)

flops_list = []
params_list = []
acc_list = []

flops, params = profile(model, inputs=(inputs, ))
flops_list.append(flops)
params_list.append(params)
flops, params = clever_format([flops, params], "%.4f")
print(flops, params)

model.cuda()
model.eval()
inputs = torch.randn(10, 3, 345, 345).cuda()

start_time = time.time()
for count in tqdm(range(0,1000)):
    model(inputs)
    count += 1

elapsed_time = time.time() - start_time
print('\n', elapsed_time/10/1000*100)

# %%
