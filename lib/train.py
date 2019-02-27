import os
import time
import sys

import torch.optim as optim
from torchnet.logger import VisdomLogger, VisdomPlotLogger
import torch.nn as nn
import torch.nn.functional as F
import torch
import visdom
import numpy as np

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def compute_loss(output, target):
    class_loss = (target * F.relu(0.9 - output)+ 0.5 * (1 - target) * F.relu(output - 0.1)).mean()
    return class_loss

def compute_acc(predict,target,vis):
    predict[predict>=0.7]=1
    predict[predict<=0.3]=0
    vis.image(predict[0].cpu().float().numpy(), env='show',win='predict', opts=dict(title='predict'))
    predict = predict != target
    acc = torch.sum(predict).float() / torch.numel(target.data)
    return acc
def train_epoch(model, loader,optimizer, epoch, n_epochs, ):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    model.train()
    end = time.time()
    vis=visdom.Visdom()
    for batch_index,(data,target) in enumerate(loader):


        data = data.cuda()
        target[target >= 1] = 1
        target = target.float()
        target = target.cuda()
        output = model(data)

        loss = compute_loss(output, target)

        batch_size = target.size(0)
        losses.update(loss.data, batch_size)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        vis.image(data[0].cpu().numpy(), env='show', win='img', opts=dict(title='img'))
        vis.image(target[0].cpu().float().numpy(), env='show', win='target', opts=dict(title='target'))
        acc=compute_acc(output.detach(),target,vis)
        accs.update(acc)
        batch_time.update(time.time() - end)
        end = time.time()
        res = '\t'.join([
            'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
            'Batch: [%d/%d]' % (batch_index, len(loader)),
            'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
            'Loss %.4f (%.4f)' % (losses.val, losses.avg),
            'Error %.4f (%.4f)' % (accs.val, accs.avg),
        ])
        print(res)
    return batch_time.avg, losses.avg  , accs.avg



def test_epoch(model,loader,epoch,n_epochs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # Model on eval mode
    model.eval()
    vis = visdom.Visdom()
    with torch.no_grad():
        end = time.time()
        for batch_index,(data,target) in enumerate(loader):
            data = data.cuda()
            target[target >= 1] = 1
            target = target.float()
            target = target.cuda()
            output = model(data)
            loss = compute_loss(output, target)

            batch_size = target.size(0)
            losses.update(loss.data, batch_size)
            vis.image(data[0].cpu().numpy(), env='show', win='img', opts=dict(title='img'))
            vis.image(target[0].cpu().float().numpy(), env='show', win='target', opts=dict(title='target'))
            acc = compute_acc(output, target,vis)
            accs.update(acc)
            batch_time.update(time.time() - end)
            end = time.time()
            res = '\t'.join([
                'Test',
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Batch: [%d/%d]' % (batch_index, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (accs.val, accs.avg),
            ])
            print(res)
    return batch_time.avg, losses.avg, accs.avg



def train(args, model,train_loader, decreasing_lr, wd=0.0001, momentum=0.9, ):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    vis = visdom.Visdom()
    vis.close(env=args.model)
    test_acc_logger = VisdomPlotLogger('line', env=args.model, opts={'title': 'Test Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', env=args.model, opts={'title': 'Test Loss'})
    train_acc_logger = VisdomPlotLogger('line', env=args.model, opts={'title': 'Train Accuracy'})
    train_loss_logger = VisdomPlotLogger('line', env=args.model, opts={'title': 'Train Loss'})
    lr_logger = VisdomPlotLogger('line', env=args.model, opts={'title': 'Learning Rate'})

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr,
                                                     gamma=0.1)
    best_train_loss = 10
    for epoch in range(args.nepoch):
        scheduler.step()
        
        _, train_loss,train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=args.nepoch,
        )
        _, test_loss,test_acc = test_epoch(
            loader=train_loader,
            model=model,
            epoch=epoch,
            n_epochs=args.nepoch,
        )
        if best_train_loss>train_loss:
            best_train_loss=train_loss
            print('best_loss'+str(best_train_loss))
            torch.save(model.state_dict(),args.params_name)
        print(train_loss)
        train_loss_logger.log(epoch, 1-float(train_loss))
        train_acc_logger.log(epoch,1-float(train_acc))
        test_acc_logger.log(epoch,1-float(test_acc))
        test_loss_logger.log(epoch,float(test_loss))
        lr_logger.log(epoch, optimizer.param_groups[0]['lr'])
