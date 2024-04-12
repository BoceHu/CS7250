from e2cnn import gspaces
from e2cnn import nn
import torch


def conv2d(feat_type_in, feat_type_hid, kernel_size, stride=1, groups=1, dilation=1, initialize=False):
    return nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride, dilation=dilation,
                     padding=(kernel_size - 1) // 2, groups=groups, initialize=initialize)


class BasicBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, flip=True, quotient=False, N=4, initialize=True, stride=1,
                 down_sample=None, dilation=1, kernel_size=3, norm_layer=None):
        super(BasicBlock, self).__init__()
        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                rep = r2_act.quotient_repr((None, 2))
            else:
                rep = r2_act.quotient_repr(2)
        else:
            rep = r2_act.regular_repr

        feat_type_in = nn.FieldType(r2_act, [rep] * in_planes)
        feat_type_hid = nn.FieldType(r2_act, [rep] * out_planes)

        self.conv1 = conv2d(feat_type_in, feat_type_hid, kernel_size=kernel_size, initialize=initialize,
                            dilation=dilation)

        self.conv2 = conv2d(feat_type_hid, feat_type_hid, kernel_size=kernel_size, initialize=initialize,
                            dilation=dilation)

        self.relu = nn.ReLU(feat_type_hid)

        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.down_sample is not None:
            identity = self.down_sample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet_18(torch.nn.Module):
    def __init__(self, block, layers, flip=True, quotient=False, N=4, initialize=True, insize=3):
        super(ResNet_18, self).__init__()
        self.N = N
        self.quotient = quotient
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        self.in_planes = 4

        feat_type_in = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * insize)
        feat_type_hid = nn.FieldType(self.r2_act, [self.repr] * self.in_planes)

        self.conv1 = nn.R2Conv(feat_type_in, feat_type_hid,
                               kernel_size=3, stride=1, padding=1, initialize=initialize)

        self.conv2 = nn.R2Conv(feat_type_hid, feat_type_hid,
                               kernel_size=3, stride=1, padding=1, initialize=initialize)
        self.relu = nn.ReLU(feat_type_hid)

        self.maxpool = nn.PointwiseMaxPool(feat_type_hid, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 8, layers[0], stride=1, flip=flip, N=N, initialize=initialize,
                                       quotient=quotient)
        self.layer2 = self._make_layer(block, 16, layers[1], stride=1, flip=flip, N=N, initialize=initialize,
                                       quotient=quotient, pool=True)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=1, flip=flip, N=N, initialize=initialize,
                                       quotient=quotient, pool=True)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=1, flip=flip, N=N, initialize=initialize,
                                       quotient=quotient, pool=True)

    def _make_layer(self, block, out_planes, block_num, stride=1, flip=True, N=4, initialize=True, quotient=False,
                    pool=False):
        down_sample = None
        if self.in_planes != out_planes:
            down_sample = nn.SequentialModule(
                nn.R2Conv(nn.FieldType(self.r2_act, [self.repr] * self.in_planes),
                          nn.FieldType(self.r2_act, [self.repr] * out_planes),
                          kernel_size=1, stride=stride, initialize=initialize),
            )
        if pool:
            layers = [nn.PointwiseMaxPool(nn.FieldType(self.r2_act, [self.repr] * self.in_planes), kernel_size=2,
                                          padding=0, ceil_mode=True)]
        else:
            layers = []

        layers.append(block(self.in_planes, out_planes, stride=1, down_sample=down_sample, flip=flip, N=N,
                            initialize=initialize, quotient=quotient))

        self.in_planes = out_planes

        for i in range(1, block_num):
            layers.append(
                block(self.in_planes, out_planes, stride=1, down_sample=None, flip=flip, N=N, initialize=initialize,
                      quotient=quotient))

        return torch.nn.Sequential(*layers)

    def forward(self, x):

        x = nn.GeometricTensor(x,
                               nn.FieldType(self.r2_act, x.shape[1] * [self.r2_act.trivial_repr]))  # [1, 3, 128, 128]
        x = self.conv1(x)  # [1, 32, 64, 64]
        feature_1 = self.relu(x)  # [1, 32, 64, 64]

        feature_1 = self.conv2(feature_1)
        feature_1 = self.relu(feature_1)  # [1, 32, 64, 64]
        # feature_1 = self.maxpool(feature_1)  # (1, 32, 32, 32)

        feature_2 = self.layer1(feature_1)  # (1, 64, 64, 64)

        feature_3 = self.layer2(feature_2)  # [1, 128, 32, 32]

        feature_4 = self.layer3(feature_3)  # [1, 256, 16, 16]
        feature_5 = self.layer4(feature_4)  # [1, 512, 8, 8]

        return [feature_1, feature_2, feature_3, feature_4, feature_5]


def resnet18(flip=True, N=4, initialize=True, quotient=False, insize=3):
    model = ResNet_18(BasicBlock, [2, 2, 2, 2], flip=flip, N=N, initialize=initialize, quotient=quotient, insize=insize)

    return model


def resnet34(flip=True, N=4, initialize=True, quotient=False, insize=3):
    model = ResNet_18(BasicBlock, [3, 4, 6, 3], flip=flip, N=N, initialize=initialize, quotient=quotient, insize=insize)

    return model


if __name__ == '__main__':
    # print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    x = torch.randn(1, 3, 64, 64).to('cuda')
    x_flip = x.flip([2])
    # x90 = torch.rot90(x, k=1, dims=(2, 3))

    model = resnet18(flip=True, N=4).to('cuda')
    out = model(x)
    out_flip = model(x_flip)

    out_flip_0 = torch.flip(out_flip[0].tensor, [2])
    out_flip_1 = torch.flip(out_flip[1].tensor, [2])
    out_flip_2 = torch.flip(out_flip[2].tensor, [2])
    out_flip_3 = torch.flip(out_flip[3].tensor, [2])
    out_flip_4 = torch.flip(out_flip[4].tensor, [2])

    # out90_0 = torch.rot90(out90[0].tensor, k=3, dims=(2, 3))
    # out90_1 = torch.rot90(out90[1].tensor, k=3, dims=(2, 3))
    # out90_2 = torch.rot90(out90[2].tensor, k=3, dims=(2, 3))
    # out90_3 = torch.rot90(out90[3].tensor, k=3, dims=(2, 3))
    # out90_4 = torch.rot90(out90[4].tensor, k=3, dims=(2, 3))

    print(out[-1].shape, out[-2].shape, out[-3].shape, out[-4].shape, out[-5].shape)

    # current_memory_allocated
    current_memory = torch.cuda.memory_allocated()
    print(f"Current memory allocated: {current_memory / 1024 ** 2:.2f} MB")

    # max_memory_allocated
    max_memory = torch.cuda.max_memory_allocated()
    print(f"Max memory allocated: {max_memory / 1024 ** 2:.2f} MB")

    print('mask grasp params: ', sum(p.numel()
                                     for p in model.parameters() if p.requires_grad))
