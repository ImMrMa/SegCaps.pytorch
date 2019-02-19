import torch
import torch.nn as nn
from models.capsule_net import SegCaps
from models.capsule_layer import CapsuleLayer
from models.unet import UNet
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(10,3,128,128)
y = torch.randn(N, D_out)
x=x.cuda()
# Use the nn package to define our model and loss function.
model=SegCaps()
#model=CapsuleLayer(1, 3, "conv", k=5, s=2, t_1=2, z_1=16, routing=3)
#model=UNet(2)
model.cuda()
# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(type(model.parameters()))
for t in range(500):
    a=model.named_parameters()
    for k,v in a:
        if v.grad is not None:
            print(v.grad,k)
            o=input('next')
    optimizer.zero_grad()
    y_pred = model(x)

    loss = y_pred.sum()
    print(t, loss.item())

    loss.backward()
    print(loss)
    o=input('sss')
    optimizer.step()
    y_pred = model(x)
    loss = y_pred.sum()
    print(t, loss.item())
    loss.backward()