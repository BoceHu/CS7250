import torch

from model.equi_resnet import resnet18, resnet34
from e2cnn import gspaces
from e2cnn import nn
import torch.nn.functional as F
from einops import rearrange


class cifar10net(torch.nn.Module):
    def __init__(self, num_classes=10, in_size=64, flip=False, last_quotient=False, N=8, initialize=True,
                 backbone='resnet18'):
        super(cifar10net, self).__init__()

        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if last_quotient:
            if flip:
                self.rep = self.r2_act.quotient_repr((None, 2))
            else:
                self.rep = self.r2_act.quotient_repr(2)
        else:
            self.rep = self.r2_act.regular_repr

        self.feat_type_in = nn.FieldType(self.r2_act, [self.r2_act.regular_repr] * in_size)
        self.feat_type_hid = nn.FieldType(self.r2_act, [self.r2_act.regular_repr] * in_size)

        self.conv1 = nn.R2Conv(self.feat_type_in, self.feat_type_hid, kernel_size=3,
                               padding=1, initialize=initialize)
        self.relu = nn.ReLU(self.feat_type_hid)
        self.linear = torch.nn.Linear(self.relu.out_type.size, num_classes)

        if backbone == 'resnet18':
            self.resnet = resnet18(flip=flip, N=N, initialize=initialize, quotient=False, insize=3)
        elif backbone == 'resnet34':
            self.resnet = resnet34(flip=flip, N=N, initialize=initialize, quotient=False, insize=3)

    def forward(self, x):
        out_res = self.resnet(x)

        x1, x2, x3, x4, x5 = out_res

        x = self.conv1(x5)
        x = self.relu(x)
        x = x.tensor

        b, c, w, h = x.shape
        out = F.avg_pool2d(x, (w, h))

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # B C H W

        return out, out_res


if __name__ == '__main__':
    # print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    x = torch.randn(1, 3, 64, 64).to('cuda')
    x90 = torch.rot90(x, k=1, dims=(2, 3))

    model = cifar10net(flip=False, N=8).to('cuda')
    out = model(x)
    out90 = model(x90)

    # current_memory_allocated
    current_memory = torch.cuda.memory_allocated()
    print(f"Current memory allocated: {current_memory / 1024 ** 2:.2f} MB")

    # max_memory_allocated
    max_memory = torch.cuda.max_memory_allocated()
    print(f"Max memory allocated: {max_memory / 1024 ** 2:.2f} MB")

    print('mask grasp params: ', sum(p.numel()
                                     for p in model.parameters() if p.requires_grad))
