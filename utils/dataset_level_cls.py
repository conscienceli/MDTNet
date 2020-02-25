import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import os
from utils.aug import ImgAugTransform
import pickle
import numpy as np


class Crossing_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None, for_test=False, test_image_num=100):
        self.images = []
        self.labels = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            records = f.read()
        if not for_test:
            records = records.split('\n')[:-test_image_num]
        else:
            records = records.split('\n')[-test_image_num:]
        for record in records[1:]:
            record = record.split(',')
            severity = int(record[4])
            if severity > 3 or severity < 0:
                continue
            # type_label = np.zeros((4), dtype=np.int8)
            # type_label[severity] = 1
            type_label = severity
            path = record[3].split('/')[2:]
            path = './data/Archive_cropped/' + '/'.join(path)
            self.images.append(path)
            self.labels.append(type_label)
                
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

        #for gc only
        # image = np.array(image)[:,:,1][...,np.newaxis]
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        tensor_transform = transforms.Compose([transforms.ToTensor()])
        # label = tensor_transform(label)
        label = np.array(label, dtype=np.float32)
        
        return [image, label]


def gen_train_loaders(BATCH_SIZE, NUM_WORKERS, test_image_num=100):
    csv_path = './data/patch - patch_list.csv'
  
    train_dataset = Crossing_Dataset(
        csv_path,
        transform = transforms.Compose([
            ImgAugTransform(),

            #for gc only
            # lambda x: Image.fromarray(x[:,:,0], mode='L'),

            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
        ]), 
        test_image_num=test_image_num)
    valid_dataset = Crossing_Dataset(
        csv_path,
        transform = transforms.Compose([
            transforms.ToTensor(),
        ]), 
        test_image_num=test_image_num)
    
    if os.path.exists('data/sampler_level_cls.pickle'):
        with open('data/sampler_level_cls.pickle', 'rb') as f:
            train_sampler, valid_sampler = pickle.load(f)
    else:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.ceil(0.1 * num_train))

        np.random.seed(32)
        # np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        np.random.shuffle(train_idx)
        np.random.shuffle(valid_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        with open('data/sampler_level_cls.pickle', 'wb') as f:
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


def gen_test_loaders(BATCH_SIZE, NUM_WORKERS, test_image_num=100):
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
        for_test=True,
        test_image_num=test_image_num)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    return test_loader