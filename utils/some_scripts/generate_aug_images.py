#%%
%matplotlib inline

import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
from PIL import Image
import matplotlib.pyplot as plt
import os

save_path = './results_aug_vis'
os.makedirs(save_path, exist_ok=True)

image = Image.open('data/Archive_cropped/201902/1035509_20190208_100758/1035509_20190208_100819_21_patch03.png')
image = np.array(image, dtype='uint8')[np.newaxis, ...]

augs = [
    iaa.Fliplr(1), # horizontally flip 50% of all images
    iaa.Flipud(1), # vertically flip 20% of all images
    iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        ),
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
    iaa.Affine(rotate=(-45, 45)),
    iaa.Affine(shear=(-16, 16)),
    iaa.GaussianBlur((0, 3.0)),
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Add((-10, 10), per_channel=0.5), 
    iaa.AddToHueAndSaturation((-20, 20)),
    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    ),
    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
    ]

for aug in augs:
    image_aug = Image.fromarray(aug(images=image)[0], mode='RGB')
    plt.imshow(image_aug)

    filename = str(aug)[: min(len(str(aug)), 5)]
    while os.path.exists(f'{save_path}/zfig_aug_{filename}.jpg'):
        filename += '1'
    image_aug.save(f'{save_path}/zfig_aug_{filename}.jpg')

# %%
