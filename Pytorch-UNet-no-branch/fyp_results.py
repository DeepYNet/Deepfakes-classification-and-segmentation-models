import json
import os
import random

from evaluate import evaluate
from unet import UNet

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
from PIL import Image

from torchvision import transforms

scale=1.0

file = open('../dfdc_deepfake_challenge/metadata.json')
file = json.load(file)

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


net = UNet(n_channels=3, n_classes=2, bilinear=False)
net.load_state_dict(torch.load('checkpoints/checkpoint_epoch_224x224_9.pth'))
net.eval()


true = 0
false = 0
all_imgs = os.listdir('../dfdc_deepfake_challenge/dataset_new_2/testing/crops/crops_testing/')
random.shuffle(all_imgs)
selected_imgs = all_imgs[:5000]

for file_name in selected_imgs:
    filename = '../dfdc_deepfake_challenge/dataset_new_2/testing/crops/crops_testing/'+file_name
    
    img = Image.open(filename).resize((224,224),Image.NEAREST)
    img = torch.from_numpy(preprocess(img, scale, is_mask=False))
    # img = torch.as_tensor(img.copy()).float().contiguous()?\
    img = img.unsqueeze(0)
    img = img.to(device='cpu', dtype=torch.float32)
    
    name = file_name.split('_')[0]
    
    if file[name+'.mp4']['label'] == 'REAL':
        true_out = 1.0
    else:
        true_out = 0.0
        
    pred = net(img)
    pred = torch.round(pred[0][0])
    pred = pred.detach().numpy()
    
    if pred == true_out:
        true+=1
    else:
        print(pred, true_out, file_name)
        false+=1
        
print('Correct {}, Wrong {}'.format(true, false))

