
import os
import yaml
import argparse

from glob import glob
from tqdm import tqdm
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from collections import OrderedDict
import numpy as np
from PIL import Image

from albumentations import RandomRotate90,Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as st_transforms

import losses
import archs

def main():
    val_img_ids = glob(os.path.join('inputs', 'rite', 'test','images', '*' + '.png'))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    val_transform = Compose([
        Resize(512, 512),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', 'rite','test', 'images'),
        mask_dir=os.path.join('inputs', 'rite','test', 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=1,
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)
    
    model = archs.__dict__['UNet'](1,3,False)
    model.load_state_dict(torch.load('/data1/fuzhenxin/SR/tt-sfuda/TT_SFUDA_2D/models/rite_unet/model.pth'))
    model.cuda()
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, meta in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            tensor = torch.sigmoid(output).detach().clone().cpu()
            
            img_tensor = tensor[0].squeeze(0)
            img_np = img_tensor.detach().numpy()

            img_pil = Image.fromarray((img_np*255).astype(np.uint8))

            pic_name=meta['img_id'][0]
            img_pil.save(f'./fzx_output/{pic_name}_foward.png')
            
            pbar.update(1)
        pbar.close


if __name__ == '__main__':
    main()