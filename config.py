import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C
__C.PWD = os.getcwd()
# ------------------------------TRAIN------------------------
__C.SEED = 978975  # random seed,  for reproduction
__C.DATASET = 'SHHA'  # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD

if __C.DATASET == 'UCF50':  # only for UCF50
    from datasets.UCF50.setting import cfg_data

    __C.VAL_INDEX = cfg_data.VAL_INDEX

if __C.DATASET == 'GCC':  # only for GCC
    from datasets.GCC.setting import cfg_data

    __C.VAL_MODE = cfg_data.VAL_MODE


__C.NET = 'OAI_NET_V2'  # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet
__C.METHOD = 't1_OAI_NET_V2_no_att_with_image_patch_rawvgg_pretrain'
__C.DESCRIBLE=''
__C.PRE_GCC = False  # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = 'path to model'  # path to model
__C.PRE = __C.PWD+'/vgg16.pth'

__C.RESUME = False  # contine training
__C.RESUME_PATH = './EXP/SHHA/t1_OAI_NET_V4_no_att_with_image_patch_02-04_09-58/latest_state.pth'

__C.GPU_ID = [0,1,2,3]  # sigle gpu: [0], [1] ...; multi gpus: [0,1]

# learning rate settings
__C.LR = 5e-5  # learning rate
__C.LR_DECAY = 0.995  # decay rate
__C.LR_DECAY_START = 50  # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1  # decay frequency
__C.MAX_EPOCH = 1500
__C.LR_CHANGER = 'cosann'
__C.EPOCH_DIS = 30
__C.ITER_DIS = 30
# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

__C.LAMBDA_1 = 1e-3  # SANet:0.001 CMTL 0.0001

# print
__C.PRINT_FREQ = 30

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = __C.DATASET + '/' + __C.METHOD + '_' + now


if __C.DATASET == 'UCF50':
    __C.EXP_NAME += '_' + str(__C.VAL_INDEX)

if __C.DATASET == 'GCC':
    __C.EXP_NAME += '_' + __C.VAL_MODE

__C.EXP_PATH = os.getcwd()+'/EXP/'  # the path of logs, checkpoints, and current codes

# ------------------------------VAL------------------------
__C.VAL_DENSE_START = 50
__C.VAL_FREQ = 30  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

# ------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1  # must be 1 for training images with the different sizes

# ================================================================================
# ================================================================================
# ================================================================================
