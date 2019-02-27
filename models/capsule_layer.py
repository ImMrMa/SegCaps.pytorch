import torch
import torch.nn as nn
import torch.nn.functional as F
import models.nn_ as nn_
import torch.optim as optim

class CapsuleLayer(nn.Module):
    def __init__(self, t_0,z_0, op, k, s, t_1, z_1, routing):
        super().__init__()
        self.t_1 = t_1
        self.z_1 = z_1
        self.op = op
        self.k = k
        self.s = s
        self.routing = routing
        self.convs = nn.ModuleList()
        self.t_0=t_0
        for _ in range(t_0):
            if self.op=='conv':
                self.convs.append(nn.Conv2d(z_0, self.t_1*self.z_1, self.k, self.s,padding=2,bias=False))
            else:
                self.convs.append(nn.ConvTranspose2d(z_0, self.t_1 * self.z_1, self.k, self.s,padding=2,output_padding=1))

    def forward(self, u):  # input [N,CAPS,C,H,W]
        if u.shape[1]!=self.t_0:
            raise ValueError("Wrong type of operation for capsule")
        op = self.op
        k = self.k
        s = self.s
        t_1 = self.t_1
        z_1 = self.z_1
        routing = self.routing
        N = u.shape[0]
        H_1=u.shape[3]
        W_1=u.shape[4]
        t_0 = self.t_0

        u_t_list = [u_t.squeeze(1) for u_t in u.split(1, 1)]  # 将cap分别取出来

        u_hat_t_list = []

        for i, u_t in zip(range(self.t_0), u_t_list):  # u_t: [N,C,H,W]
            if op == "conv":
                u_hat_t = self.convs[i](u_t)  # 卷积方式
            elif op == "deconv":
                u_hat_t = self.convs[i](u_t) #u_hat_t: [N,t_1*z_1,H,W]
            else:
                raise ValueError("Wrong type of operation for capsule")
            H_1 = u_hat_t.shape[2]
            W_1 = u_hat_t.shape[3]
            u_hat_t = u_hat_t.reshape(N, t_1,z_1,H_1, W_1).transpose_(1,3).transpose_(2,4)
            u_hat_t_list.append(u_hat_t)    #[N,H_1,W_1,t_1,z_1]
        v=self.update_routing(u_hat_t_list,k,N,H_1,W_1,t_0,t_1,routing)
        return v
    def update_routing(self,u_hat_t_list, k, N, H_1, W_1, t_0, t_1, routing):
        one_kernel = torch.ones(1, t_1, k, k).cuda()  # 不需要学习
        b = torch.zeros(N, H_1, W_1, t_0, t_1).cuda()  # 不需要学习
        b_t_list = [b_t.squeeze(3) for b_t in b.split(1, 3)]
        u_hat_t_list_sg = []
        for u_hat_t in u_hat_t_list:
            u_hat_t_sg=u_hat_t.detach()
            u_hat_t_list_sg.append(u_hat_t_sg)

        for d in range(routing):
            if d < routing - 1:
                u_hat_t_list_ = u_hat_t_list_sg
            else:
                u_hat_t_list_ = u_hat_t_list

            r_t_mul_u_hat_t_list = []
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                # routing softmax (N,H_1,W_1,t_1)
                b_t.transpose_(1, 3).transpose_(2, 3)  #[N,t_1,H_1, W_1]
                b_t_max = torch.nn.functional.max_pool2d(b_t,k,1,padding=2)
                b_t_max = b_t_max.max(1, True)[0]
                c_t = torch.exp(b_t - b_t_max)
                sum_c_t = nn_.conv2d_same(c_t, one_kernel, stride=(1, 1))  # [... , 1]
                r_t = c_t / sum_c_t  # [N,t_1, H_1, W_1]
                r_t = r_t.transpose(1, 3).transpose(1, 2)  # [N, H_1, W_1,t_1]
                r_t = r_t.unsqueeze(4)  # [N, H_1, W_1,t_1, 1]
                r_t_mul_u_hat_t_list.append(r_t * u_hat_t)  # [N, H_1, W_1, t_1, z_1]
            p = sum(r_t_mul_u_hat_t_list)  # [N, H_1, W_1, t_1, z_1]
            v = squash(p)
            if d < routing - 1:
                b_t_list_ = []
                for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                    # b_t     : [N, t_1,H_1, W_1]
                    # u_hat_t : [N, H_1, W_1, t_1, z_1]
                    # v       : [N, H_1, W_1, t_1, z_1]
                    # [N,H_1,W_1,t_1]
                    b_t.transpose_(1,3).transpose_(2,1)
                    b_t_list_.append(b_t + (u_hat_t * v).sum(4))
        v.transpose_(1, 3).transpose_(2, 4)
        # print(v.grad)
        return v
    def squash(self, p):
        p_norm_sq = (p * p).sum(-1, True)
        p_norm = (p_norm_sq + 1e-9).sqrt()
        v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
        return v


def update_routing(u_hat_t_list,k,N,H_1,W_1,t_0,t_1,routing):
    one_kernel = torch.ones(1, t_1, k, k).cuda()#不需要学习
    b = torch.zeros(N, H_1, W_1, t_0, t_1 ).cuda()#不需要学习
    b_t_list = [b_t.squeeze(3) for b_t in b.split(1, 3)]
    u_hat_t_list_sg = []
    for u_hat_t in u_hat_t_list:
        u_hat_t_sg = u_hat_t.clone()
        u_hat_t_sg.detach_()
        u_hat_t_list_sg.append(u_hat_t_sg)

    for d in range(routing):
        if d < routing - 1:
            u_hat_t_list_ = u_hat_t_list_sg
        else:
            u_hat_t_list_ = u_hat_t_list

        r_t_mul_u_hat_t_list = []
        for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
            # routing softmax (N,H_1,W_1,t_1)
            b_t.transpose_(1, 3).transpose_(2, 3)
            torch.nn.functional.max_pool2d(b_t,k,)
            b_t_max = nn_.max_pool2d_same(b_t, k, 1)
            b_t_max = b_t_max.max(1, True)[0]
            c_t = torch.exp(b_t - b_t_max)
            sum_c_t = nn_.conv2d_same(c_t, one_kernel, stride=(1, 1))  # [... , 1]
            r_t = c_t / sum_c_t  # [N,t_1, H_1, W_1]
            r_t = r_t.transpose(1, 3).transpose(1, 2)  # [N, H_1, W_1,t_1]
            r_t = r_t.unsqueeze(4)  # [N, H_1, W_1,t_1, 1]
            r_t_mul_u_hat_t_list.append(r_t * u_hat_t)  # [N, H_1, W_1, t_1, z_1]

        p = sum(r_t_mul_u_hat_t_list)  # [N, H_1, W_1, t_1, z_1]
        v = squash(p)
        if d < routing - 1:
            b_t_list_ = []
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                # b_t     : [N, t_1,H_1, W_1]
                # u_hat_t : [N, H_1, W_1, t_1, z_1]
                # v       : [N, H_1, W_1, t_1, z_1]
                b_t = b_t.transpose(1, 3).transpose(1, 2)  # [N,H_1,W_1,t_1]
                b_t_list_.append(b_t + (u_hat_t * v).sum(4))
            b_t_list = b_t_list_
        v.transpose_(1,3).transpose_(2,4)
    #print(v.grad)
    return v

def squash( p):
    p_norm_sq = (p * p).sum(-1, True)
    p_norm = (p_norm_sq + 1e-9).sqrt()
    v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
    return v

def test():
    m=CapsuleLayer(1, 16, "conv", k=5, s=1, t_1=2, z_1=16, routing=1)
    #m=cheng_layer()
    m=m.cuda()
    b=input('s')
    a=torch.randn(10, 1, 16, int(b), int(b))
    #a=torch.randn(2,2)
    a=a.cuda()
    optimizer = optim.Adam(m.parameters(), lr=1)
    for k,v in m.named_parameters():
        print(k)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20],
                                               gamma=0.1)
    b=m(a)
    c=b.mean()
    for k in m.parameters():
        print(k)
    print(b.shape)
    print(c)
    c.backward()
    optimizer.step()
    b=m(a)
    c=b.mean()
    print(c)
    print(a.grad)
    print(b.shape)
    #print(a[1, :, :, 1, 1])

#test()
def test1():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    a = []

    b = torch.ones([1, 10, 10, 2, 3]).cuda()
    print(b)
    a.append(b)
    c = update_routing(a, 2, 1, 10, 10, 1, 2,3)
    print(c.cpu().numpy())

#test1()
