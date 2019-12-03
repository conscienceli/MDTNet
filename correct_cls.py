# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from utils import dataset, train
from utils.dataset_correct import gen_train_loaders
from torchvision.models import resnet50
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

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

loss_func = F.binary_cross_entropy_with_logits
if not os.path.exists('best_state_correct.pth'):
    model = train.train_model('correct', model, dataloaders_all, device, optimizer_ft, loss_func, exp_lr_scheduler,  num_epochs=20000)
else:
    model.load_state_dict(torch.load('best_state_correct.pth'))
    model = train.train_model('correct', model, dataloaders_all, device, optimizer_ft, loss_func, exp_lr_scheduler, num_epochs=20000)


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
model.load_state_dict(torch.load('best_state_correct.pth'))
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

for record in records:
    record = record.split(',')
    severity = int(record[4])
    type_label = np.ones((1))
    if severity > 3 or severity < 0:
        type_label[0] = 0
    path = record[3].split('/')[1:]
    path = './data/' + '/'.join(path)
    images.append(path)
    labels.append(type_label)

corrects = 0.
tp = 0.
tn = 0.
fp = 0.
fn = 0.
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
    preds = nn.functional.sigmoid(preds)
    print(preds.item(), labels[image_id])

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

    raw_image.save(f'results_correct/ROW({image_id+records_number-test_range+1:04})-PR({preds.item():.2})-GT({labels[image_id]})-{isCorrect}.png')

    model = model.cpu()
    vis_cam(device, model, raw_image, 0, f'ROW({image_id+records_number-test_range+1:04})-PR({preds.item():.2})-GT({labels[image_id]})-{isCorrect}', directory='./results_correct')

print('accuracy: ', corrects/ float(len(images)))
print('precision: ', tp/(tp+fp))
print('recall: ', tp/(tp+fn))
    # break



# %%
