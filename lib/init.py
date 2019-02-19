import os
import torch
from models import *
from lib.dataloader import get_data
def init(args):

    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")
    os.environ['CUDA_VISIBLE_DEVICES'] =args.gpu
    print('{}:{}'.format('cuda',torch.cuda.is_available()))
    args.cuda = torch.cuda.is_available() #查看cuda是否正常
    torch.manual_seed(args.seed)#没看懂是在干什么,应该是随机生成数
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loader = get_data(args.batch_size_train,data_name=args.data_name,data_root='../data_test')
    #if args.model=='xception':
    if 'xception' in args.model :
        model=Xception()
    elif 'densenet' in args.model:
        model=DenseNet101()
    elif 'shufflenet' in args.model:
        model=ShuffleNet(10, g = 8, scale_factor = 1)
    elif 'deformnet' in args.model:
        model=DeformConvNet()
    elif 'unet' in args.model:
        model=UNet(6)
        if args.pretrain:
            model.load_state_dict(torch.load(args.load_params_name))
    elif 'segcaps' in args.model:
        model=SegCaps()
    model.cuda()
    
    decreasing_lr = list(map(int,args.dlr.split(',')))
    return  train_loader,model,decreasing_lr