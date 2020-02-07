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


def vis_cam(device, model, raw_image, target_class, target_layer, file_name_to_export, directory='./results'):
    image = np.array(raw_image)
    original_image = raw_image
    transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    image = transform(image).unsqueeze(0)

    pretrained_model = model

    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=target_layer)
    # Generate cam mask
    cam = grad_cam.generate_cam(image, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export, directory)