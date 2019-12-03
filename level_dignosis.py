# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from utils import dataset, train
from torchvision.models import resnet50
import torch
import torch.nn as nn
import os
from utils.dataset import gen_train_loaders

model = resnet50(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

BATCH_SIZE = 96
NUM_WORKERS = 8
dataloaders_all = gen_train_loaders(BATCH_SIZE, NUM_WORKERS)

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-4)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

loss_func = nn.MSELoss(size_average=False)
if not os.path.exists('best_state_level.pth'):
    model = train.train_model('level', model, dataloaders_all, device, optimizer_ft, loss_func, exp_lr_scheduler, num_epochs=20000)
else:
    model.load_state_dict(torch.load('best_state_level.pth'))
    model = train.train_model('level', model, dataloaders_all, device, optimizer_ft, loss_func, exp_lr_scheduler, num_epochs=20000)


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
import sklearn.metrics

model = resnet50(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
model.load_state_dict(torch.load('best_state_level.pth'))
model.eval()   # Set model to evaluate mode


images = []
labels = []

csv_path = './data/patch - patch_list.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    records = f.read()
records_number = len(records.split('\n'))
test_range = 100
# test_range = records_number-1
records = records.split('\n')[-test_range:]
rows_id = []
row_id = 1

for record in records:
    row_id += 1
    record = record.split(',')
    severity = int(record[4])
    if severity > 3 or severity < 0:
        continue
    path = record[3].split('/')[1:]
    path = './data/' + '/'.join(path)
    images.append(path)
    labels.append(severity)
    rows_id.append(row_id)

label_avg = sum(labels) / len(labels)
mse = 0
vari = 0
Y_true = []
Y_pred = []
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
    if preds.item() < 0.:
        pred = 0.
    elif preds.item() > 3.:
        pred = 3.
    else:
        pred = preds.item()
    print(pred, labels[image_id])

    Y_true.append(labels[image_id])
    Y_pred.append(pred)

    mse = mse + (pred - labels[image_id])*(pred - labels[image_id])
    vari = vari + (pred - label_avg)*(pred- label_avg)

    isCorrect = 'True'
    if labels[image_id] == 0:
        if pred >= 0.5:
            isCorrect = 'Wrong'
    elif labels[image_id] == 3:
        if pred < 2.5:
            isCorrect = 'Wrong'
    else:
        if abs(pred- labels[image_id]) > 0.5:
            isCorrect = 'Wrong'

    raw_image.save(f'results/ROW({rows_id[image_id]:04})-PR({pred:.2})-GT({labels[image_id]})-{isCorrect}.png')

    model = model.cpu()
    vis_cam(device, model, raw_image, 0, f'ROW({rows_id[image_id]:04})-PR({pred:.2})-GT({labels[image_id]})-{isCorrect}')

    # break
mse = mse / len(labels)
vari = vari / len(labels)
print(mse, vari)

print(sklearn.metrics.r2_score(Y_true, Y_pred))
print(sklearn.metrics.mean_squared_error(Y_true, Y_pred))
print(sklearn.metrics.explained_variance_score(Y_true, Y_pred))

# %%
