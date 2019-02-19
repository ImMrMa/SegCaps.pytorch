import torch
import torch.nn as nn
from models.capsule_layer import CapsuleLayer
import models.nn_

class SegCaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 16, 5,1,padding=2)
        self.step_1 = nn.Sequential(  # 1/2
            CapsuleLayer(1,16, "conv", k=5, s=2, t_1=2, z_1=16, routing=3),
            CapsuleLayer(2,16, "conv", k=5, s=1, t_1=4, z_1=16, routing=3)
        )
        self.step_2 = nn.Sequential(  # 1/4
            CapsuleLayer(2,16, "conv", k=5, s=2, t_1=4, z_1=32, routing=3),
            CapsuleLayer(4,32, "conv", k=5, s=1, t_1=4, z_1=32, routing=3)
        )
        self.step_3 = nn.Sequential(  # 1/8
            CapsuleLayer(4,32, "conv", k=5, s=2, t_1=8, z_1=32, routing=3),
            CapsuleLayer(8,32, "conv", k=5, s=1, t_1=8, z_1=32, routing=3)
        )
        self.step_4 = CapsuleLayer(8,32, "deconv", k=5, s=2, t_1=8, z_1=32, routing=3)
        self.step_5 = CapsuleLayer(8,32, "deconv", k=5, s=1, t_1=4, z_1=32, routing=3)
        self.step_6 = CapsuleLayer(4,32, "deconv", k=5, s=2, t_1=4, z_1=16, routing=3)
        self.step_7 = CapsuleLayer(4,16, "deconv", k=5, s=1, t_1=4, z_1=16, routing=3)
        self.step_8 = CapsuleLayer(4,16, "deconv", k=5, s=2, t_1=2, z_1=16, routing=3)
        self.step_9 = CapsuleLayer(2,16, "deconv", k=5, s=1, t_1=1, z_1=16, routing=3)

    def forward(self, x):
        x = self.conv_1(x)
        x.unsqueeze_(1)
        skip_1 = x
        x = self.step_1(x)
        skip_2 = x
        x = self.step_2(x)

        skip_3 = x

        x = self.step_3(x)

        x = self.step_4(x)
        x=torch.cat((x,skip_3),1)
        x = self.step_5(x)

        x=self.step_6(x)
        x = torch.cat((x, skip_2), 1)
        x=self.step_7(x)
        x=self.step_8(x)
        x=torch.cat((x,skip_1),1)
        x=self.step_9(x)

        x.squeeze_(1)
        print(x.shape)
        v_lens = self.compute_vector_length(x)
        print(v_lens.shape)
        #x = self.conv2d(x, 64, 1)
        #x = self.conv2d(x, 128, 1)
        #recons = self.conv2d(x, images.get_shape()[-1], 1)
        return v_lens

    def compute_vector_length(self, x):
        out=(x*x+ 1e-9).sum(1,True)
        out.sqrt_()
        return out
def test():
    model=SegCaps()
    model=model.cuda()
    print(model)
    c=input('s')
    a=torch.randn(10,3,int(c),int(c))
    a=a.cuda()
    b=model(a)
    print(b.shape)
#test()