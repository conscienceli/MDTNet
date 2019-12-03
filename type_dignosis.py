# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from utils import dataset, train
from utils.dataset_type import gen_train_loaders
from torchvision.models import resnet50
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

model = resnet50(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=15, bias=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

BATCH_SIZE = 96
NUM_WORKERS = 8
dataloaders_all = gen_train_loaders(BATCH_SIZE, NUM_WORKERS)

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-4)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

loss_func = F.cross_entropy
if not os.path.exists('best_state_type.pth'):
    model = train.train_model('type', model, dataloaders_all, device, optimizer_ft, loss_func, exp_lr_scheduler,  num_epochs=20000)
else:
    model.load_state_dict(torch.load('best_state_type.pth'))
    model = train.train_model('type', model, dataloaders_all, device, optimizer_ft, loss_func, exp_lr_scheduler, num_epochs=20000)


# %%
from utils import dataset, train
from torchvision.models import resnet50
import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from utils.vis_cam import vis_cam

model = resnet50(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
model.load_state_dict(torch.load('best_state_type.pth'))
model.eval()   # Set model to evaluate mode


images = []
labels = []

csv_path = './data/patch - patch_list.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    records = f.read()
records = records.split('\n')[-100:]

for record in records[1:]:
    record = record.split(',')
    severity = int(record[4])
    if severity > 3 or severity < 0:
        continue
    path = record[3].split('/')[1:]
    path = './data/' + '/'.join(path)
    images.append(path)
    labels.append(severity)

label_avg = sum(labels) / len(labels)
mse = 0
vari = 0
for image_id, image in enumerate(images):
    image = Image.open(image)
    image = np.array(image)[1422:1422+345, 150:150+345,:]
    raw_image = Image.fromarray(image, mode='RGB')
    # image = image[np.newaxis,...]
    # image = Image.fromarray(image, mode='RGB')
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    model = model.to(device)
    preds = model(image)
    print(preds.item(), labels[image_id])

    mse = mse + (preds.item() - labels[image_id])*(preds.item() - labels[image_id])
    vari = vari + (preds.item() - label_avg)*(preds.item() - label_avg)

    raw_image.save(f'results/ID({image_id})-PR({preds.item():.2})-GT({labels[image_id]}).png')

    model = model.cpu()
    vis_cam(device, model, raw_image, 0, f'ID({image_id})-PR({preds.item():.2})-GT({labels[image_id]})')

    # break
mse = mse / len(labels)
vari = vari / len(labels)
print(mse, vari)
