import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random

import logging
from os import listdir
from os.path import splitext
from pathlib import Path

images_dir = Path('../dfdc_deepfake_challenge/dataset/crops/')
masks_dir = Path('../dfdc_deepfake_challenge/dataset/masks/')

ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]

def load(filename):
    ext = splitext(filename)[1]
    if ext in ['.npz', '.npy']:
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename).resize((224,320))
        
def preprocess(pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
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
    
    
arr_to_save_img = []
arr_to_save_mask = []

count = 0
for name in ids[:10]:
    count += 1
    mask_file = list(masks_dir.glob(name + '.*'))
    img_file = list(images_dir.glob(name + '.*'))
    
    assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
    assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        
    mask = load(mask_file[0])
    img = load(img_file[0])
    
    assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

    img = preprocess(img, 1.0, is_mask=False)
    mask = preprocess(mask, 1.0, is_mask=True)
    
    arr_to_save_img.append(img)
    arr_to_save_mask.append(mask)
    print("[STATUS] Count:  ", count)
    
np.save('img_array.npy', np.array(arr_to_save_img))
np.save('mask_array.npy', np.array(arr_to_save_mask))