import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from config import cfg
from termcolor import cprint

class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name, loss_1_fn, loss_2_fn,pretrained= None):
        super(CrowdCounter, self).__init__()

        if model_name == 'SANet':
            from models.M2TCC_Model.SANet import SANet as net
        if model_name == 'OAI_NET_V4':
            from models.M2TCC_Model.OAINet import OAI_NET_V4 as net


        self.CCN = net()
        
        
        if pretrained:

            if 'SHA' in pretrained:
                cprint('update parameter from SHA pretrain model mae %.3f' % temp_mae, color='yellow')
                check = torch.load(pretrained,map_location=torch.device('cpu'))
                temp_mae = check['best_mae']
                pretrained_dict = check['net_state_dict']
                model_dict = self.CCN.state_dict()  # 自己的模型参数变量
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k[9:] in model_dict}  # 去除一些不需要的参数
                model_dict.update(pretrained_dict)  # 参数更新
                self.CCN.load_state_dict(model_dict)  # 加载
                
            else:
                
                cprint('update parameter from imagenet pretrain model', color='yellow')
                pretrained_dict = torch.load(pretrained,map_location=torch.device('cpu'))['state_dict']
                model_dict = self.CCN.state_dict()  # 自己的模型参数变量
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                                   k[7:] in model_dict}  # only update backbone
                model_dict.update(pretrained_dict)  # 参数更新
                self.CCN.load_state_dict(model_dict)  # 加载
           
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()

        self.loss_1_fn = loss_1_fn.cuda()
        self.loss_2_fn = loss_2_fn.cuda()

    @property
    def loss(self):
        return self.loss_1, self.loss_2 * cfg.LAMBDA_1

    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        self.loss_1 = self.loss_1_fn(density_map.squeeze(), gt_map.squeeze())
        self.loss_2 = 1 - self.loss_2_fn(density_map, gt_map[:, None, :, :])

        return density_map

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map
