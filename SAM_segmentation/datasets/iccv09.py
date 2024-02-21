import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity

"""
class_names,r,g,b
sky,68,1,84
tree,72,40,140
road,62,74,137
grass,38,130,142
water,31,158,137
building,53,183,121
mountain,109,205,89
foreground,180,222,44
unknown,49,104,142
"""

mean = [0.4813, 0.4901, 0.4747] # rgb
std = [0.2495, 0.2492, 0.2748] # rgb

class Iccv2009Dataset(data.Dataset):
    
    rgb2id = {
        (68, 1, 84): 0,
        (72, 40, 140): 1,
        (62, 74, 137): 2,
        (38, 130, 142): 3,
        (31, 158, 137): 4,
        (53, 183, 121): 5,
        (109, 205, 89): 6,
        (180, 222, 44): 7,
        (49, 104, 142): 8,
    }
    
    def __init__(self, root, split, transform=None):
        
        self.image_root = os.path.join(root, 'images')
        self.mask_root = os.path.join(root, 'labels_colored')
        self.split = split
        self.images = []
        self.targets = []
        self.transform = transform
        
        for filename in os.listdir(self.image_root):
            if filename.endswith('.jpg'):
                self.images.append(os.path.join(self.image_root, filename))
                self.targets.append(os.path.join(self.mask_root, filename[:-4] + '.png'))
                
        if self.split == 'train':
            self.images = self.images[:int(0.7*len(self.images))]
            self.targets = self.targets[:int(0.7*len(self.targets))]
        elif self.split == 'val':
            self.images = self.images[int(0.7*len(self.images)):]
            self.targets = self.targets[int(0.7*len(self.targets)):]
        else:
            raise ValueError('Invalid split name: {}'.format(self.split))
        
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        target = self.encode_mask(np.array(target))
        target = Image.fromarray(target)
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        # tensor min-max normalization image, type(image) = Tensor
        image = (image - image.min())/(image.max() - image.min())

        return image, target
    
    def __len__(self):
        return len(self.images)
    
    @classmethod
    def encode_mask(cls, mask):
        for k in cls.rgb2id:
            mask[(mask == k).all(axis=2)] = cls.rgb2id[k]
        return mask[:, :, 0]
    
    @classmethod
    def decode_target(cls, target):
        target_rgb = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
        for k in cls.rgb2id:
            target_rgb[(target == cls.rgb2id[k])] = k
        return target_rgb
    
if __name__ == "__main__":
    dataset = Iccv2009Dataset('/mnt/nasv2/hhz/DeepLabV3Plus-Pytorch-master/datasets/data/iccv09', 'train')
    # test mask shape and value
    for i in range(len(dataset)):
        img, mask = dataset[i]
        img = np.array(img)
        mask = np.array(mask)
        print(img.shape, mask.shape)
        print(np.unique(mask))
        if i == 10:
            break
        