import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .SHHA import SHHA
from .setting import cfg_data 
import torch
import random
import pdb
import torch.nn.functional as F



def get_min_size(batch):

    min_ht = cfg_data.TRAIN_SIZE[0]
    min_wd = cfg_data.TRAIN_SIZE[1]
    dis = cfg_data.DIVISIBLE
    
    for i_sample in batch:
        
        _,ht,wd = i_sample.shape
        if ht<min_ht:
            min_ht = ht
        if wd<min_wd:
            min_wd = wd
            
    min_ht = min_ht//dis*dis
    min_wd = min_wd//dis*dis
    
    return min_ht,min_wd

def random_crop(img,den,dst_size):
    # dst_size: ht, wd

    _,ts_hd,ts_wd = img.shape
    
    
    x1 = random.randint(0, ts_wd - dst_size[1])//cfg_data.LABEL_FACTOR*cfg_data.LABEL_FACTOR
    y1 = random.randint(0, ts_hd - dst_size[0])//cfg_data.LABEL_FACTOR*cfg_data.LABEL_FACTOR
    x2 = x1 + dst_size[1]
    y2 = y1 + dst_size[0]

    label_x1 = x1//cfg_data.LABEL_FACTOR
    label_y1 = y1//cfg_data.LABEL_FACTOR
    label_x2 = x2//cfg_data.LABEL_FACTOR
    label_y2 = y2//cfg_data.LABEL_FACTOR

    return img[:,y1:y2,x1:x2], den[label_y1:label_y2,label_x1:label_x2]

def share_memory(batch):
    out = None
    if False:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out

def SHHA_crop_collate(batch):
    # @GJY 
    r"""Puts each data field into a tensor with outer dimension batch size"""

    transposed = list(zip(*batch)) # imgs and dens
    imgs, dens = [transposed[0],transposed[1]]


    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):
        
        min_ht, min_wd = get_min_size(imgs)
        
        
        cropped_imgs = []
        cropped_dens = []
        for i_sample in range(len(batch)):
            _img, _den = random_crop(imgs[i_sample],dens[i_sample],[min_ht,min_wd])
            cropped_imgs.append(_img)
            cropped_dens.append(_den)


        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_dens = torch.stack(cropped_dens, 0, out=share_memory(cropped_dens))

        return [cropped_imgs,cropped_dens]

    raise TypeError((error_msg.format(type(batch[0]))))

    
def SHHA_raw_collate(batch):
    # @GJY 
    r"""Puts each data field into a tensor with outer dimension batch size"""

    transposed = list(zip(*batch)) # imgs and dens
    imgs, dens = [transposed[0],transposed[1]]

    dis = cfg_data.DIVISIBLE
    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):
        reshaped_imgs = []
        reshaped_dens = []
        for i_sample in range(len(batch)):
            c,ih,iw = imgs[i_sample].shape
            dh,dw = dens[i_sample].shape
            
            assert ih==dh and iw==dw
            
            nh = ih//dis*dis
            nw = iw//dis*dis
            zoom = iw*ih/nh/nw
            
            
            new_img = imgs[i_sample]
            new_img = new_img.unsqueeze(0)
            new_img = F.interpolate(new_img, (nh, nw), mode='bilinear', align_corners=False)
            new_img = new_img.squeeze(0)
            
            new_den = dens[i_sample]
            new_den = new_den.unsqueeze(0)
            new_den = new_den.unsqueeze(0)
            
            new_den = F.interpolate(new_den, (nh, nw), mode='bilinear', align_corners=False)*zoom
            new_den = new_den.squeeze(0)
            new_den = new_den.squeeze(0)
            
            reshaped_imgs.append(new_img)
            reshaped_dens.append(new_den)


        reshaped_imgs = torch.stack(reshaped_imgs, 0, out=share_memory(reshaped_imgs))
        reshaped_dens = torch.stack(reshaped_dens, 0, out=share_memory(reshaped_dens))

        return [reshaped_imgs,reshaped_dens]

    raise TypeError((error_msg.format(type(batch[0]))))


def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    factor = cfg_data.LABEL_FACTOR
    train_main_transform = own_transforms.Compose([
    	own_transforms.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.GTScaleDown(factor),
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = SHHA(cfg_data.DATA_PATH+'/train_data', mode = 'train',preload = True,main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    
    train_loader =None
    
    if cfg_data.TRAIN_BATCH_SIZE==1:
        train_loader = DataLoader(train_set, batch_size=1, num_workers=8, collate_fn=SHHA_raw_collate,shuffle=True, drop_last=True)
        
    elif cfg_data.TRAIN_BATCH_SIZE>1:
        train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8, collate_fn=SHHA_crop_collate, shuffle=True, drop_last=True)
    
    

    val_set = SHHA(cfg_data.DATA_PATH+'/test_data', mode = 'test',preload = True, main_transform=None, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, collate_fn=SHHA_raw_collate,shuffle=True, drop_last=False)

    return train_loader, val_loader, restore_transform
