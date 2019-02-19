import argparse
from lib.init import init
from lib.train import train





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Little Ma's train")
    parser.add_argument('--batch_size_train', type=int, default=8, help='input batch size for training (default: 160)')
    parser.add_argument('--batch_size_test', type=int, default=60, help='input batch size for testing (default: 80)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--gpu', default='1', help='index of gpus to use')
    parser.add_argument('--dlr', default='30,60', help='decreasing strategy')
    parser.add_argument('--model', default='segcaps-test', help='which model (default: xception)')
    parser.add_argument('--data_root', default='./', help='data_root (default: ./)')
    parser.add_argument('--nepoch', type=int,default=110, help='epochs (default: 200)')
    parser.add_argument('--seed', type=int,default='1', help='seed (default: 1)')
    parser.add_argument('--pretrain', type=int, default='0', help='pretrain (default: 1)')
    parser.add_argument('--data_name', default='train', help='data_name (default: my_ai_000)')
    parser.add_argument('--params_name', default='params_test.pkl', help='params_name (default: params.pkl)')
    parser.add_argument('--load_params_name', default='params.pkl', help='params_name (default: params.pkl)')
    args = parser.parse_args()
    train_loader,model,decreasing_lr=init(args)
    train(args,model,train_loader,
            decreasing_lr,wd=0.0001, momentum=0.9)
    print('hhh')
    print('Done!')


