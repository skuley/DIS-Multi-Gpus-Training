import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
import cv2
import numpy as np
import albumentations as A

from tqdm import tqdm


class Dataset(Dataset):
    def __init__(self, image_path='../../data/DIS5K/DIS-TR/im', gt_path='../../data/DIS5K/DIS-TR/gt',
                 image_transform=None, gt_transform=None, load_on_mem=False):
        self.images = sorted(glob(os.path.join(image_path, '*.jpg')))
        self.gts = sorted(glob(os.path.join(gt_path, '*.png')))
        
        self.image_transform = image_transform
        self.gt_transform = gt_transform
        
        print(f'images : {len(self.images)}')
        print(f'gts : {len(self.gts)}')
        
        self.load_on_mem = load_on_mem

        if self.load_on_mem:
            self.load_data()

    def __len__(self):
        return len(self.gts)

    def load_data(self):
        self.im_lst = []
        self.gt_lst = []
        for im, gt in tqdm(zip(self.images, self.gts), total=self.__len__()):
            image, gt = cv2.imread(im), cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            self.im_lst.append(image)
            self.gt_lst.append(gt)

    def _transform(self, image, gt, image_transform=None, gt_transform=None):
        if self.image_transform:
            transformed = image_transform(image=image)
            image = transformed['image']
        if self.gt_transform:
            transformed = gt_transform(image=image, mask=gt)
            image, gt = transformed['image'], transformed['mask']

        image = transforms.ToTensor()(image)
        gt = transforms.ToTensor()(gt)
        return image, gt

    def __getitem__(self, idx):
        if self.load_on_mem:
            image, gt = self.im_lst[idx], self.gt_lst[idx]
        else:
            image, gt = cv2.imread(self.images[idx]), cv2.imread(self.gts[idx], cv2.IMREAD_GRAYSCALE)
        image, gt = self._transform(image, gt,
                                    image_transform=self.image_transform, 
                                    gt_transform=self.gt_transform)

        return {'image': image, 'gt': gt}