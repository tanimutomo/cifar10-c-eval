import numpy as np
import os
import PIL
import torch
import torchvision

from PIL import Image
from torchvision import datasets

from utils import load_txt

corruptions = load_txt('./corruptions.txt')


class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str, level :int,
                 transform=None, target_transform=None):
        assert name in corruptions
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        label_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.label = np.load(label_path)
        
    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return img, label
    
    def __len__(self):
        return len(self.data)


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)