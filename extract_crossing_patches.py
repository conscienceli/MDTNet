import os
from PIL import Image
import numpy as np

cropped_path = 'data/Archive_cropped/'
os.makedirs(cropped_path, exist_ok=True)

paths = []
with open('data/patch - patch_list.csv', 'r', encoding='utf-8') as f:
    records = f.read().split('\n')
    for record in records[1:]:
        record = record.split(',')
        path = record[3].split('/')[1:]
        path = '/'.join(path)
        paths.append(path)

for path in paths:
    image = Image.open('data/' + path)
    image = np.array(image)[1422:1422+345, 150:150+345,:]
    image = Image.fromarray(image, mode='RGB')

    directory = cropped_path + '/'.join(path.split('/')[:-1])
    os.makedirs(directory, exist_ok=True)
    image.save(cropped_path + path)
