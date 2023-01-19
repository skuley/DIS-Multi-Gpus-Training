from PIL import Image, ImageOps
import os
import os.path as osp
from torch.utils.data import Dataset
from glob import glob
import cv2
import numpy as np
from torchvision.transforms import transforms as T
from tqdm import tqdm


class Dataset(Dataset):
    def __init__(self, image_path='', transform=None):
        super(Dataset, self).__init__()
#         assert osp.isdir(image_path) == True, f"{image_path} is not a directory"

        self.image_path = image_path

        self.images = sorted(glob(osp.join(image_path, '*.png')))
        
        print(f'images : {len(self.images)}')

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def _transform(self, image: str):
        image = Image.open(image)
        if self.transform:
            image = ImageOps.grayscale(image)
            image = np.array(image)
            sample = self.transform(image=image)
            image = sample['image']
        
        image = T.ToTensor()(image)
        return image

    def __getitem__(self, item):
        image = self.images[item]
        image = self._transform(image)
        return {'image': image, 'gt': image}