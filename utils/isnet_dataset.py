import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
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
        
        
        self.load_on_mem = load_on_mem

        if self.load_on_mem:
            self.load_data_on_mem()

    def __len__(self):
        return len(self.gts)

    def load_data_on_mem(self):
        self.im_lst = []
        self.gt_lst = []
        for im, gt in tqdm(zip(self.images, self.gts), total=self.__len__()):
            image, gt = Image.open(im).convert("RGB"), Image.open(gt).convert('L')
            image, gt = np.array(image), np.array(gt)
            if self.gt_transform:
                transformed = self.gt_transform[0](image=image, mask=gt)
                image, gt = transformed['image'], transformed['mask']
            self.im_lst.append(image)
            self.gt_lst.append(gt)

    def _transform(self, image, gt, image_transform=None, gt_transform=None):
        if self.image_transform:
            transformed = image_transform(image=image)
            image = transformed['image']
        if self.gt_transform:
            transformed = gt_transform(image=image, mask=gt)
            image, gt = transformed['image'], transformed['mask']

        # image = transforms.ToTensor()(image)
        # gt = transforms.ToTensor()(gt)
        return image, gt

    def __getitem__(self, idx):
        if self.load_on_mem:
            image, gt = self._transform(self.im_lst[idx], self.gt_lst[idx],
                                        image_transform=self.image_transform,
                                        gt_transform=A.Compose(self.gt_transform[1:]))
        else:
            image, gt = Image.open(self.images[idx]).convert("RGB"), Image.open(self.gts[idx]).convert('L')
            image, gt = np.array(image), np.array(gt)
            image, gt = self._transform(image, gt,
                                        image_transform=self.image_transform, 
                                        gt_transform=self.gt_transform)

        return {'image': image, 'gt': gt}