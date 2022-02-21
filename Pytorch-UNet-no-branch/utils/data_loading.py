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
        self.scale = scale
        
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        random.shuffle(self.ids)
        self.ids = self.ids
        
        self.file = open('../dfdc_deepfake_challenge/metadata.json')
        self.file = json.load(self.file)
#         print(self.file)


        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
            
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        #newW, newH = int(scale * w), int(scale * h)
        newW,newH = 224,224
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img_ndarray = np.asarray(pil_img)


        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

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
        name = self.ids[idx]
        vdo_name = name.split('_')[0]
        if self.file[vdo_name+'.mp4']['label'] == 'REAL':
            true_out = np.array([1])
        else:
            true_out = np.array([0])
        
        try:
            img_file = '../dfdc_deepfake_challenge/dataset_new_1/training/crops/'+name+'.png'

            img = self.load(img_file)
        except:
            img_file = '../dfdc_deepfake_challenge/dataset_new_1/testing/crops/crops_testing/'+name+'.png'

            img = self.load(img_file)

        img = self.preprocess(img, self.scale, is_mask=False)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'output': torch.as_tensor(true_out.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
