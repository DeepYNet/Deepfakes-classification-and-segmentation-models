import argparse
import logging
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

from utils.dice_score import multiclass_dice_coeff, dice_coeff

def benchmark(model):
    model.eval()
    #test_dir = os.listdir('data/imgs')
    test_dir = os.listdir('fyp_data/crops')
    #mask_dir = os.listdir('data/masks')
    test_dir.sort()
    mask_dir = os.listdir('fyp_data/masks')
    mask_dir.sort()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    dice_score = 0
    count = 0
    save_path = 'prediction'


    for idx,(i,mask) in enumerate(zip(test_dir,mask_dir)):
        #filename = 'data/imgs/'+i
        filename = 'fyp_data/crops/'+i
        mask_filename = 'fyp_data/masks/'+mask
        mask_img = Image.open(mask_filename)
        total_img = 0
        total_img += idx
        img = Image.open(filename)
        img = torch.from_numpy(BasicDataset.preprocess(img, 1, is_mask=False))
        #print(img.shape)
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            # predict the mask
            mask_pred = net(img)
            #print(mask_pred)
            total_img +=1
            if mask_pred >0.5:
                #print(' ture image prediction made on image = ',filename)
                #print('mask is at = ',mask_filename)
                mask_img.save('prediction/'+mask)

                count +=1

            # convert to one-hot format
            
    print('total count = ',count)
    print('total image = ',total_img)


net = UNet(n_channels=3, n_classes=2,bilinear=False)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

net.load_state_dict(torch.load('checkpoints/checkpoint_epoch_224x224_99.pth'))
   # net.eval()
net.to(device)


time1 = time.time()
print(benchmark(net))
torch.cuda.synchronize()
time2 = time.time()
print('time taken = ',time2-time1)
