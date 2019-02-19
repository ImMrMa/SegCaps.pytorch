import argparse
from lib.init import init
import torch
import visdom
import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Little Ma's train")
    parser.add_argument('--batch_size_train', type=int, default=1, help='input batch size for training (default: 160)')
    parser.add_argument('--batch_size_test', type=int, default=60, help='input batch size for testing (default: 80)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--gpu', default='5', help='index of gpus to use')
    parser.add_argument('--dlr', default='0,30', help='decreasing strategy')
    parser.add_argument('--model', default='unet', help='which model (default: xception)')
    parser.add_argument('--data_root', default='./', help='data_root (default: ./)')
    parser.add_argument('--nepoch', type=int, default=150, help='epochs (default: 200)')
    parser.add_argument('--seed', type=int, default='1', help='seed (default: 1)')
    parser.add_argument('--pretrain', type=int, default='1', help='pretrain (default: 1)')
    parser.add_argument('--data_name', default='train_rotate', help='data_name (default: my_ai_000)')
    parser.add_argument('--params_name', default='params.pkl', help='params_name (default: params.pkl)')
    parser.add_argument('--load_params_name', default='params_rotate.pkl', help='params_name (default: params.pkl)')
    args = parser.parse_args()
    train_loader,model,decreasing_lr=init(args)
    model.eval()
    vis=visdom.Visdom()
    for batch_index,(data,target) in enumerate(train_loader):
        
        data=data.cuda()
        target=target.cuda()
        end = time.time()
        output = model(data)
        predict = output.data
        max = torch.max(predict, 1, True)[0]
        predict = predict == max
        predict = predict.float()
        predict=predict[0,:,:,:]
        predict=predict[0,:,:].unsqueeze(0).expand(3,-1,-1)*torch.tensor([0,0,0]).reshape(3,1,1).float().cuda()+predict[2,:,:].unsqueeze(0).expand(3,-1,-1)*torch.tensor([255,215,0]).reshape(3,1,1).float().cuda()+predict[3,:,:].unsqueeze(0).expand(3,-1,-1)*torch.tensor([255,193,193]).reshape(3,1,1).float().cuda()+predict[1,:,:].unsqueeze(0).expand(3,-1,-1)*torch.tensor([255,218,185]).reshape(3,1,1).float().cuda()+predict[4,:,:].unsqueeze(0).expand(3,-1,-1)*torch.tensor([69,139,116]).reshape(3,1,1).float().cuda()+predict[5,:,:].unsqueeze(0).expand(3,-1,-1)*torch.tensor([205,85,85]).reshape(3,1,1).float().cuda()
        vis.image(
            data[0],
            env='test',
            win='source',
            opts=dict(title='source')
        )
        vis.image(
            predict,
            env='test',
            win='predict',
            opts=dict(title='predict')
        )
        vis.image(
            target.float()/5,
            env='test',
            win='label',
            opts=dict(title='label')
        )
        a=input('ss')

    print('hhh')
    print('Done!')
