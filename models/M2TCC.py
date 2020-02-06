import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from config import cfg
from termcolor import cprint


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name, loss_1_fn, loss_2_fn, pretrained=None):
        super(CrowdCounter, self).__init__()

        if model_name == 'SANet':
            from models.M2TCC_Model.SANet import SANet as net
        if model_name == 'OAI_NET_V4':
            from models.M2TCC_Model.OAINet import OAI_NET_V4 as net
        if model_name == 'OAI_NET_V6':
            from models.M2TCC_Model.OAINet import OAI_NET_V6 as net
        if model_name == 'OAI_NET_V2':
            from models.M2TCC_Model.OAINet import OAI_NET_V2 as net

        self.CCN = net()

        if pretrained:

            if 'SHA' in pretrained:

                check = torch.load(pretrained, map_location=torch.device('cpu'))
                temp_mae = check['best_mae']
                cprint('update parameter from SHA pretrain model mae %.3f' % temp_mae, color='yellow')
                pretrained_dict = check['net_state_dict']
                model_dict = self.CCN.state_dict()  # 自己的模型参数变量
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k[9:] in model_dict}  # 去除一些不需要的参数
                model_dict.update(pretrained_dict)  # 参数更新
                self.CCN.load_state_dict(model_dict)  # 加载

            elif 'imagenet' in pretrained:

                cprint('update parameter from imagenet pretrain model', color='yellow')
                pretrained_dict = torch.load(pretrained, map_location=torch.device('cpu'))['state_dict']
                model_dict = self.CCN.state_dict()  # 自己的模型参数变量
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                                   k[7:] in model_dict}  # only update backbone
                model_dict.update(pretrained_dict)  # 参数更新
                self.CCN.load_state_dict(model_dict)  # 加载

            elif 'can' in pretrained:

                pre = torch.load(pretrained, map_location=torch.device('cpu'))
                
                self.CCN.backbone.Stage1.center_branch.C1.conv.weight.data = pre['frontend.0.weight']
                self.CCN.backbone.Stage1.center_branch.C1.conv.bias.data = pre['frontend.0.bias']
                self.CCN.backbone.Stage1.center_branch.C2.conv.weight.data = pre['frontend.2.weight']
                self.CCN.backbone.Stage1.center_branch.C2.conv.bias.data = pre['frontend.2.bias']

                self.CCN.backbone.Stage2.center_branch.C1.conv.weight.data = pre['frontend.5.weight']
                self.CCN.backbone.Stage2.center_branch.C1.conv.bias.data = pre['frontend.5.bias']
                self.CCN.backbone.Stage2.center_branch.C2.conv.weight.data = pre['frontend.7.weight']
                self.CCN.backbone.Stage2.center_branch.C2.conv.bias.data = pre['frontend.7.bias']

                self.CCN.backbone.Stage3.center_branch.C1.conv.weight.data = pre['frontend.10.weight']
                self.CCN.backbone.Stage3.center_branch.C1.conv.bias.data = pre['frontend.10.bias']
                self.CCN.backbone.Stage3.center_branch.C2.conv.weight.data = pre['frontend.12.weight']
                self.CCN.backbone.Stage3.center_branch.C2.conv.bias.data = pre['frontend.12.bias']
                self.CCN.backbone.Stage3.center_branch.C3.conv.weight.data = pre['frontend.14.weight']
                self.CCN.backbone.Stage3.center_branch.C3.conv.bias.data = pre['frontend.14.bias']

                self.CCN.backbone.Stage4[0].conv.weight.data = pre['frontend.17.weight']
                self.CCN.backbone.Stage4[0].conv.bias.data = pre['frontend.17.bias']
                self.CCN.backbone.Stage4[1].conv.weight.data = pre['frontend.19.weight']
                self.CCN.backbone.Stage4[1].conv.bias.data = pre['frontend.19.bias']
                self.CCN.backbone.Stage4[2].conv.weight.data = pre['frontend.21.weight']
                self.CCN.backbone.Stage4[2].conv.bias.data = pre['frontend.21.bias']
                cprint('update parameter from can vggbackbone', color='yellow')

            elif 'vgg' in pretrained:
                
                pre = torch.load(pretrained, map_location=torch.device('cpu'))
                self.CCN.backbone.Stage1.center_branch.C1.conv.weight.data = pre['features.0.weight']
                self.CCN.backbone.Stage1.center_branch.C1.conv.bias.data = pre['features.0.bias']
                self.CCN.backbone.Stage1.center_branch.C2.conv.weight.data = pre['features.2.weight']
                self.CCN.backbone.Stage1.center_branch.C2.conv.bias.data = pre['features.2.bias']

                self.CCN.backbone.Stage2.center_branch.C1.conv.weight.data = pre['features.5.weight']
                self.CCN.backbone.Stage2.center_branch.C1.conv.bias.data = pre['features.5.bias']
                self.CCN.backbone.Stage2.center_branch.C2.conv.weight.data = pre['features.7.weight']
                self.CCN.backbone.Stage2.center_branch.C2.conv.bias.data = pre['features.7.bias']

                self.CCN.backbone.Stage3.center_branch.C1.conv.weight.data = pre['features.10.weight']
                self.CCN.backbone.Stage3.center_branch.C1.conv.bias.data = pre['features.10.bias']
                self.CCN.backbone.Stage3.center_branch.C2.conv.weight.data = pre['features.12.weight']
                self.CCN.backbone.Stage3.center_branch.C2.conv.bias.data = pre['features.12.bias']
                self.CCN.backbone.Stage3.center_branch.C3.conv.weight.data = pre['features.14.weight']
                self.CCN.backbone.Stage3.center_branch.C3.conv.bias.data = pre['features.14.bias']

                self.CCN.backbone.Stage4[0].conv.weight.data = pre['features.17.weight']
                self.CCN.backbone.Stage4[0].conv.bias.data = pre['features.17.bias']
                self.CCN.backbone.Stage4[1].conv.weight.data = pre['features.19.weight']
                self.CCN.backbone.Stage4[1].conv.bias.data = pre['features.19.bias']
                self.CCN.backbone.Stage4[2].conv.weight.data = pre['features.21.weight']
                self.CCN.backbone.Stage4[2].conv.bias.data = pre['features.21.bias']
                cprint('update parameter from raw imagenet vgg backbone', color='yellow')

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
