import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    final_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        count = 0
        image, true_out = batch['image'], batch['output']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        true_out = true_out.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            pred = net(image)
            print(pred, true_outtrue_out)
            for i in range(0, len(pred)):
                if torch.ceil(pred[i][0]) == torch.ceil(true_out[i][0]):
                    count += 1
        
        final_score += count/len(batch)

           

    net.train()
    return final_score / num_val_batches
