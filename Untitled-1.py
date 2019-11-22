# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%


# %%
from utils import dataset, train
from torchvision.models import resnet50
import torch
import torch.nn as nn
import os

model = resnet50(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

BATCH_SIZE = 96
NUM_WORKERS = 8

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-4)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

if not os.path.exists('best_state.pth'):
    model = train.train_model(model, device, optimizer_ft, exp_lr_scheduler, BATCH_SIZE, NUM_WORKERS, num_epochs=20000)
else:
    model.load_state_dict(torch.load('best_state.pth'))
    # model = train.train_model(model, device, optimizer_ft, exp_lr_scheduler, BATCH_SIZE, NUM_WORKERS, num_epochs=20000)



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
model.load_state_dict(torch.load('best_state.pth'))
model.eval()   # Set model to evaluate mode


images = []
labels = []

csv_path = './data/patch - patch_list.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    records = f.read()
records = records.split('\n')[-100:]

for record in records[1:]:
    record = record.split(',')
    sevirity = int(record[4])
    if sevirity > 3 or sevirity < 0:
        continue
    path = record[3].split('/')[1:]
    path = './data/' + '/'.join(path)
    images.append(path)
    labels.append(sevirity)

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


# %%

from utils import dataset, train
from torchvision.models import resnet50
import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from utils.vis.cnn_layer_visualization import CNNLayerVisualization

model = resnet50(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
model.load_state_dict(torch.load('best_state.pth'))

cnn_layer = 7
filter_pos = 5
# Fully connected layer is not needed
image = Image.open('./data/Archive/201310/1006690_20131028_110252/1006690_20131028_110252_48_patch01.png')
image = np.array(image)[1422:1422+345, 150:150+345,:]
layer_vis = CNNLayerVisualization(model, cnn_layer, filter_pos, image)

# Layer visualization with pytorch hooks
layer_vis.visualise_layer_with_hooks()

# %%
from utils import dataset, train
from torchvision.models import resnet50
import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from utils.vis.gradcam import GradCam
from utils.vis.misc_functions import save_class_activation_images

model = resnet50(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
model.load_state_dict(torch.load('best_state.pth'))

image = Image.open('./data/Archive/201310/1006690_20131028_110252/1006690_20131028_110252_48_patch01.png')
image = np.array(image)[1422:1422+345, 150:150+345,:]
original_image = Image.fromarray(image, mode='RGB')
transform = transforms.Compose([
            transforms.ToTensor(),
        ])
image = transform(image).unsqueeze(0)

prep_img = image
target_class = 0
file_name_to_export = '1006690_20131028_110252_48_patch01'
pretrained_model = model

# Grad cam
grad_cam = GradCam(pretrained_model, target_layer=7)
# Generate cam mask
cam = grad_cam.generate_cam(prep_img, target_class)
# Save mask
save_class_activation_images(original_image, cam, file_name_to_export)
original_image.save('results/1006690_20131028_110252_48_patch01.png')
print('Grad cam completed')

# %%
