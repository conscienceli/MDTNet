import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import os
from utils.aug import ImgAugTransform
import pickle
import numpy as np


class Crossing_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None, for_test=False):
        self.images = []
        self.labels = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            records = f.read()
        if not for_test:
            records = records.split('\n')[:-100]
        else:
            records = records.split('\n')[-100:]
        for record in records[1:]:
            record = record.split(',')
            sevirity = int(record[4])
            if sevirity > 3 or sevirity < 0:
                continue
            path = record[3].split('/')[1:]
            path = './data/' + '/'.join(path)
            self.images.append(path)
            self.labels.append(sevirity)
                
        assert(len(self.images) == len(self.labels))
        print('data_num: ',len(self.images))
#         print(self.images)
#         print(self.labels)
                
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):        
        image = self.images[idx]
        image = Image.open(image)
        image = np.array(image)[1422:1422+345, 150:150+345,:]
        image = Image.fromarray(image, mode='RGB')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        tensor_transform = transforms.Compose([transforms.ToTensor()])
        # label = tensor_transform(label)
        label = np.array([label], dtype=np.float32)
        
        return [image, label]


def gen_train_loaders(BATCH_SIZE, NUM_WORKERS):
    csv_path = './data/patch - patch_list.csv'
  
    train_dataset = Crossing_Dataset(
        csv_path,
        transform = transforms.Compose([
            ImgAugTransform(),
            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
        ]))
    valid_dataset = Crossing_Dataset(
        csv_path,
        transform = transforms.Compose([
            transforms.ToTensor(),
        ]))
    
    if os.path.exists('data/sampler.pickle'):
        with open('data/sampler.pickle', 'rb') as f:
            train_sampler, valid_sampler = pickle.load(f)
    else:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))

        np.random.seed(32)
        # np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        np.random.shuffle(train_idx)
        np.random.shuffle(valid_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        with open('data/sampler.pickle', 'wb') as f:
            pickle.dump([train_sampler, valid_sampler],f)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    return (train_loader, valid_loader)


def gen_test_loaders(BATCH_SIZE, NUM_WORKERS):
    csv_path = './data/patch - patch_list.csv'
    
#     # Data loading code
#     traindir = os.path.join(path, 'train')
#     valdir = os.path.join(path, 'val')
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
    test_dataset = Crossing_Dataset(
        csv_path,
        transform = transforms.Compose([
            transforms.ToTensor(),
        ]),
        for_test=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    return test_loader