import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import json


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        random.shuffle(self.ids)
        self.ids = self.ids
        
        self.file = open('../dfdc_deepfake_challenge/metadata.json')
        self.file = json.load(self.file)

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
#         print("At preprocess")
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        #print('before resize')
        pil_img = pil_img.resize((newW, newH))
        #print('after resize')
        img_ndarray = np.asarray(pil_img)
        #print('speed test')
        #print('at img_ndarray = ',img_ndarray)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255
#         print("Done Preprocess")

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename).resize((224,224),Image.NEAREST)

    def __getitem__(self, idx):
#         print("Get item")
        name = self.ids[idx]
        vdo_name = name.split('_')[0]
        if self.file[vdo_name+'.mp4']['label'] == 'REAL':
            true_out = np.array([1])
        else:
            true_out = np.array([0])
        
        try:
            mask_file = '../dfdc_deepfake_challenge/dataset_new_1/training/masks/'+name+'.gif'
            img_file = '../dfdc_deepfake_challenge/dataset_new_1/training/crops/'+name+'.png'

            mask = self.load(mask_file)
            img = self.load(img_file)
        except:
            mask_file = '../dfdc_deepfake_challenge/dataset_new_1/testing/masks/masks_testing/'+name+'.gif'
            img_file = '../dfdc_deepfake_challenge/dataset_new_1/testing/crops/crops_testing/'+name+'.png'

            mask = self.load(mask_file)
            img = self.load(img_file)

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'output': torch.as_tensor(true_out.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
