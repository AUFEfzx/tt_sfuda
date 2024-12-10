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

cudnn.benchmark = True

def my_save(ps_output, pic_name, val):
    img_tensor = ps_output.cpu()[0][val]
    img_np = img_tensor.numpy()
    img_pil = Image.fromarray((img_np*255).astype(np.uint8))
    img_pil.save(pic_name)

def my_save_rgb(ps_output, pic_name, val):

    img_tensor = ps_output.squeeze(0).permute(1, 2, 0).cpu()
    img_np = img_tensor.numpy()
    img_np = img_np[:,:,::-1]
    img_pil = Image.fromarray((img_np*255).astype(np.uint8)) 
    img_pil.save(pic_name)

    # img_tensor = ps_output.cpu()[0][val]
    # img_np = img_tensor.numpy()
    # img_pil = Image.fromarray((img_np*255).astype(np.uint8))
    # img_pil.save(pic_name)

def gt2mult_tensor(tensor, classes = 2):
    __mask = tensor.cpu().numpy().astype(np.uint8)[0][0]
    _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
    _mask[__mask > 200] = 255
    _mask[(__mask > 50) & (__mask < 201)] = 128

    __mask[_mask == 0] = 2
    __mask[_mask == 255] = 0
    __mask[_mask == 128] = 1

    mask = np.zeros((__mask.shape[0], __mask.shape[1], classes))
    mask[__mask == 1] = [0, 1]
    mask[__mask == 2] = [1, 1]
    mask = np.array(mask).astype(np.uint8).transpose((2, 0, 1))
    ret_t = torch.from_numpy(mask).float()

    return ret_t.unsqueeze(0).cuda()

def out2img(tensor):
    tensor_in = torch.sigmoid(tensor)
    ret_map = np.zeros([tensor.shape[2], tensor.shape[3]], dtype=np.int8)
    ret_map[:,:]=255
    #0,0 1,0 0
    #0,1 128
    #1,1 0
    ret_map[tensor_in[0, 1] == 1] = 128
    ret_map[tensor_in[1, 1] == 0] = 0
    return torch.from_numpy(ret_map)


def train_src_model():
    #dataset path /data1/fuzhenxin/SR/SFDA_DPL/Fundus/Domain3/train/ROIs
    train_img_ids = glob(os.path.join('inputs', 'REFUGE', 'train','image', '*' + 'png'))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]
    
    train_transform = Compose([
        Resize(320, 320),
        transforms.Normalize(),
    ])
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs','REFUGE' , 'train','image'),
        mask_dir=os.path.join('inputs', 'REFUGE', 'train','mask'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=1,
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=3,
        drop_last=True)
    
    print("Creating model UNet...!!!")
    print("Loading source trained model...!!!")
    src_model = archs.__dict__['UNet'](2,3,False)
    # src_model.load_state_dict(torch.load('models/%s/model.pth'%config['name']))
    src_model.cuda()
    src_model.train()
    criterion = losses.__dict__['BCEDiceLoss']().cuda()

    src_params = filter(lambda p: p.requires_grad, src_model.parameters())
    src_optimizer = optim.Adam(src_params, lr=1.0e-05, weight_decay=0.0001)

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    for epoch in range(100):
        pbar = tqdm(total=len(train_loader))
        for input, target, path in train_loader:
            # my_save(target, "fzx.png", 0)
            # import pdb
            # pdb.set_trace()

            w_input = input.cuda()
            target = target.cuda()

            src_optimizer.zero_grad()
            output = src_model(w_input)
            
            loss_seg1 = criterion(output[0,0,...].cuda(), gt2mult_tensor(target)[0,0,...].cuda())
            loss_seg2 = criterion(output[0,1,...].cuda(), gt2mult_tensor(target)[0,1,...].cuda())
            loss = loss_seg1 + loss_seg2
            loss.backward()
            src_optimizer.step()
            # import pdb
            # pdb.set_trace()
            # avg_meters['iou'].update(iou, input.size(0))
            avg_meters['loss'].update(loss_seg1.item(), input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    
    torch.save(src_model.state_dict(), './fzx_output/fzx_REFUGE_ttsfuda_src.pth')
    
def validate_src_model():
    val_img_ids = glob(os.path.join('inputs', 'REFUGE', 'test','image', '*' + 'png'))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    val_transform = Compose([
        Resize(320, 320),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', 'REFUGE','test', 'image'),
        mask_dir=os.path.join('inputs', 'REFUGE','test', 'mask'),
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
    
    model = archs.__dict__['UNet'](2,3,False)
    model.load_state_dict(torch.load('/data1/fuzhenxin/SR/tt-sfuda/TT_SFUDA_2D/models/REFUGE/fzx_REFUGE_ttsfuda_src.pth'))
    model.cuda()
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        val_cup_dice = 0.0
        val_disc_dice = 0.0
        datanum_cnt = 0.0
        avg_meters = {'val_cup_dice': AverageMeter(),
                  'val_disc_dice': AverageMeter()}
        for input, target, meta in val_loader:
            input = input.cuda()
            target = target.cuda()
            pic_name=meta['img_id'][0]
            my_save_rgb(input, f'./fzx_output/{pic_name}_input_src.png', 0)

            output = model(input)
            dice_cup, dice_disc = dice_coeff_2label(output, gt2mult_tensor(target))
            
            import pdb
            pdb.set_trace()
            my_save(target, f'./fzx_output/{pic_name}_gt_src.png', 0)
            my_save(output, f'./fzx_output/{pic_name}_0_foward.png', 0)
            my_save(output, f'./fzx_output/{pic_name}_1_foward.png', 1)
            my_save(gt2mult_tensor(target), f'./fzx_output/{pic_name}_0_gt.png', 0)
            my_save(gt2mult_tensor(target), f'./fzx_output/{pic_name}_1_gt.png', 1)
            criterion = losses.__dict__['BCEDiceLoss']().cuda()
            loss_seg1 = criterion(output[0,0,...].cuda(), gt2mult_tensor(target)[0,0,...].cuda())
            loss_seg2 = criterion(output[0,1,...].cuda(), gt2mult_tensor(target)[0,1,...].cuda())
            loss = loss_seg1 + loss_seg2
            print(f"fffzx loss {loss.item()}")
            


            # import pdb
            # pdb.set_trace()
            # output = postprocessing(output.cpu())
            
            # tensor = torch.sigmoid(output).detach().clone().cpu()
            
            # import pdb
            # pdb.set_trace()
            # img_np = out2img(output).detach().numpy()

            # img_pil = Image.fromarray((img_np).astype(np.uint8))
            # new_size = (800, 800)
            # img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)

            # pic_name=meta['img_id'][0]
            # img_pil.save(f'./fzx_output/{pic_name}_0_foward.png')

            # img_tensor = (output.detach().clone().cpu())[0][1].squeeze(0)
            # img_np = img_tensor.detach().numpy()

            # img_pil = Image.fromarray((img_np*255).astype(np.uint8))
            # new_size = (800, 800)
            # img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)

            # pic_name=meta['img_id'][0]
            # img_pil.save(f'./fzx_output/{pic_name}_1_foward.png')

            val_cup_dice = np.sum(dice_cup)
            val_disc_dice = np.sum(dice_disc)
            
            # import pdb
            # pdb.set_trace()
            # datanum_cnt += float(dice_cup.shape[0])
            # val_cup_dice /= datanum_cnt
            # val_disc_dice /= datanum_cnt

            avg_meters['val_cup_dice'].update(val_cup_dice, input.size(0))
            avg_meters['val_disc_dice'].update(val_disc_dice, input.size(0))
            
            postfix = OrderedDict([
                ('val_cup_dice', avg_meters['val_cup_dice'].getlast()),
                ('val_disc_dice', avg_meters['val_disc_dice'].getlast()),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    print("Done...")

if __name__ == '__main__':
    # train_src_model()


    np.bool = np.bool_
    import os
    import sys
    path = '/data1/fuzhenxin/SR/SFDA_DPL/SFDA-DPL/utils/'
    sys.path.insert(0, path)
    from metrics1 import dice_coeff_2label
    from Utils import postprocessing

    validate_src_model()