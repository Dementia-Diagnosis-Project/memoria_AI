# utils.py (PyTorch version)

import random
import numpy as np
import torch
import os
import pandas as pd
import collections
import re
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fold_path(data_path):
    fold = collections.defaultdict(list)
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)):
            fold[folder] = os.listdir(os.path.join(data_path, folder))
    return fold

def create_dataset(data, labels):
    data_list = []
    for idx, row in data.iterrows():
        for num in row['Image_Number']:
            data_list.append({
                'image_path': os.path.join(row['image_path'], f"sharp2_plane{num}.png"),
                'Gender': row['Gender'],
                'Age_Group': row['Age_Group'],
                'image_number': num,
                'label': labels[idx]
            })
    return pd.DataFrame(data_list)

class MRIDataset(Dataset):
    def __init__(self, dataframe, augment=False):
        self.dataframe = dataframe
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.augment_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        if self.augment:
            image = self.augment_transform(image)
        else:
            image = self.transform(image)

        gender = torch.tensor(row['Gender']).long()
        age_group = torch.tensor(row['Age_Group']).long()
        image_number = torch.tensor(row['image_number']).long()
        label = torch.tensor(row['label']).long()

        return (image, (gender, age_group, image_number)), label

def make_dataloader(df, batch_size=30, shuffle=True, augment=False):
    dataset = MRIDataset(df, augment=augment)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return loader