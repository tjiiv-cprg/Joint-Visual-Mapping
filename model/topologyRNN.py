import torch
torch.set_default_tensor_type(torch.FloatTensor)
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


# Returns 2D convolutional layer with space-preserving padding
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
    if transposed:
        layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1, dilation=dilation, bias=bias)
        # Bilinear interpolation init
        w = torch.Tensor(kernel_size, kernel_size)
        centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
        for y in range(kernel_size):
            for x in range(kernel_size):
                w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs((y - centre) / stride))
        layer.weight.data.copy_(w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    if bias:
        init.constant(layer.bias, 0)
    return layer


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.inplanes = inplanes
        if self.stride != 1 or inplanes !=planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Residual(nn.Module):

    def __init__(self, kernel_size, inplanes, planes, stride):
        super(Residual, self ).__init__()
        # self.insnorm1 = nn.InstanceNorm2d(inplanes)
        # self.relu = nn.ReLU(inplace=True)
        self.resblock1 = ResBasicBlock(inplanes, planes, stride=stride)

        # self.insnorm2 = nn.InstanceNorm2d(planes)
        # self.resblock2 = ResBasicBlock(planes, planes, stride=1)

    def forward(self, x):
        out1 = self.resblock1(x)
        # out2 = self.resblock2(self.relu(self.insnorm2(out1)))
        return out1

class IRC(nn.Module):

    def __init__(self, kernel_size, inplanes, planes, stride):
        super(IRC, self ).__init__()
        self.insnorm = nn.InstanceNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.inplanes = inplanes
        self.conv = conv(self.inplanes, planes, kernel_size=kernel_size, stride=stride)

    def forward(self,x):
        out = self.relu(self.insnorm(self.conv(x)))

        return out


class IRUC(nn.Module):

    def __init__(self, kernel_size, inplanes, planes, stride):
        super(IRUC, self ).__init__()
        self.insnorm = nn.InstanceNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = inplanes
        self.upsample = nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None)
        self.conv = conv(self.inplanes, planes, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        out = self.relu(self.insnorm(self.conv(self.upsample(x))))
        return out


class GlobalFeature(nn.Module):

    def __init__(self):
        super(GlobalFeature, self ).__init__()
        self.conv1 = conv(in_planes=1, out_planes=16, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res1_1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=2)
        # self.res1_2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=2)
        self.res1_2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)

        # self.res2_1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res2_1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=2)
        self.res2_2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)

        self.res3_1 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res3_2 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        self.res4_1 = Residual(kernel_size=3, inplanes=32, planes=64, stride=2)
        self.res4_2 = Residual(kernel_size=3, inplanes=64, planes=64, stride=1)

        self.res5_1 = Residual(kernel_size=3, inplanes=64, planes=128, stride=2)
        self.res5_2 = Residual(kernel_size=3, inplanes=128, planes=128, stride=1)

        self.irc1_1 = IRC(kernel_size=1, inplanes=128, planes=64, stride=1)
        self.iruc1 = IRUC(kernel_size=1, inplanes=64, planes=64, stride=1)
        self.irc1_2 = IRC(kernel_size=1, inplanes=64, planes=64, stride=1)

        self.irc2_1 = IRC(kernel_size=1, inplanes=64, planes=32, stride=1)
        self.iruc2 = IRUC(kernel_size=1, inplanes=32, planes=32, stride=1)
        self.irc2_2 = IRC(kernel_size=1, inplanes=32, planes=32, stride=1)

        self.irc3_1 = IRC(kernel_size=1, inplanes=32, planes=16, stride=1)
        self.iruc3 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.irc3_2 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)

        self.irc4_1 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.iruc4 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.irc4_2 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)

        self.iruc5 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.conv5 = conv(in_planes=16, out_planes=8, kernel_size=1, stride=1)

    def forward(self, x):
        x0 = self.maxpool(self.conv1(x))
        x1 = self.res1_2(self.res1_1(x0))
        x2 = self.res2_2(self.res2_1(x1))
        x3 = self.res3_2(self.res3_1(x2))
        x4 = self.res4_2(self.res4_1(x3))
        x5 = self.res5_2(self.res5_1(x4))
        y4 = self.irc1_2(self.iruc1(self.irc1_1(x5)))
        y4 += x4
        y3 = self.irc2_2(self.iruc2(self.irc2_1(y4)))
        y3 += x3
        y2 = self.irc3_2(self.iruc3(self.irc3_1(y3)))
        y2 += x2
        y1 = self.irc4_2(self.iruc4(self.irc4_1(y2)))

        y1 += x1

        y0 = self.conv5(self.iruc5(y1))

        return y0

class DistanceTransform(nn.Module):

    def __init__(self):
        super(DistanceTransform, self ).__init__()
        self.conv1 = conv(in_planes=8, out_planes=16, kernel_size=1, stride=1)
        self.res1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res3 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res4 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.conv2 = conv(in_planes=16, out_planes=1, kernel_size=1, stride=1)

        self.net = nn.Sequential(self.conv1, self.res1, self.res2, self.res3, self.res4, self.conv2)

    def forward(self, x):
        out = self.net(x)
        out[out > 10] = 10
        out[out < 0] = 0
        return out



class ConvRNNCell(nn.Module):
    def __init__(self, input_c, hidden_c, kernel_size):
        """
        input_shape: (channel, h, w)
        hidden_c: the number of hidden channel.
        kernel_shape: (h, w)
        """
        super(ConvRNNCell, self ).__init__()
        self.input_c = input_c
        self.hidden_c = hidden_c
        self.kernel_size = kernel_size

        self.conv_input2hidden = conv(in_planes=self.input_c, out_planes=self.hidden_c, kernel_size=self.kernel_size, stride=1)

        self.conv_hidden = conv(in_planes=2*self.hidden_c, out_planes=self.hidden_c, kernel_size=self.kernel_size, stride=1)

        self.insnorm1 = nn.InstanceNorm2d(self.hidden_c)
        self.insnorm2 = nn.InstanceNorm2d(self.hidden_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_cur, hidden_prev):
        input_conv = self.relu(self.insnorm1(self.conv_input2hidden(input_cur)))
        hidden_concat = torch.cat([input_conv, hidden_prev], dim=1) # (batch, c, h, w)
        hidden_cur = self.relu(self.insnorm2(self.conv_hidden(hidden_concat)))

        return hidden_cur


class ConvLSTMCell(nn.Module):
    def __init__(self, input_c, hidden_c, kernel_size, pad_mod='replicate'):
        """
        input_shape: (channel, h, w)
        hidden_c: the number of hidden channel.
        kernel_shape: (h, w)
        """
        super().__init__()
        self.input_c = input_c
        self.hidden_c = hidden_c
        self.kernel_size = kernel_size

        self.gate_input_conv = nn.Conv2d(in_channels=self.input_c + self.hidden_c,
                                   out_channels=self.hidden_c,
                                   kernel_size=self.kernel_size,
                                   stride=1,
                                   padding=self.padding_size,
                                   padding_mode=pad_mod)
        self.gate_output_conv = nn.Conv2d(in_channels=self.input_c + self.hidden_c,
                                   out_channels=self.hidden_c,
                                   kernel_size=self.kernel_size,
                                   stride=1,
                                   padding=self.padding_size,
                                   padding_mode=pad_mod)
        self.gate_forget_conv = nn.Conv2d(in_channels=self.input_c + self.hidden_c,
                                   out_channels=self.hidden_c,
                                   kernel_size=self.kernel_size,
                                   stride=1,
                                   padding=self.padding_size,
                                   padding_mode=pad_mod)
        self.gate_cell_conv = nn.Conv2d(in_channels=self.input_c + self.hidden_c,
                                   out_channels=self.hidden_c,
                                   kernel_size=self.kernel_size,
                                   stride=1,
                                   padding=self.padding_size,
                                   padding_mode=pad_mod)
        self.norm = nn.GroupNorm(self.hidden_c, self.hidden_c)

        # self.gate_forget_conv.weight.data.normal_(0, 0.02)
        nn.init.constant_(self.gate_forget_conv.bias, 1.0)

        nn.init.orthogonal_(self.gate_cell_conv.weight, gain=1)

    def forward(self, input_cur, state_prev):
        hidden_prev, cell_prev = state_prev
        input_concat = torch.cat([input_cur, hidden_prev], dim=1)  # (batch, c, h, w)'

        gate_input = self.norm(self.gate_input_conv(input_concat))
        gate_forget = self.norm(self.gate_forget_conv(input_concat))
        gate_output = self.norm(self.gate_output_conv(input_concat))
        gate_cell = self.norm(self.gate_cell_conv(input_concat))

        gate_input = torch.sigmoid(gate_input)
        gate_forget = torch.sigmoid(gate_forget)
        gate_output = torch.sigmoid(gate_output)
        gate_cell = torch.tanh(gate_cell)

        cell_cur = (gate_forget * cell_prev) + (gate_input * gate_cell)
        hidden_cur = torch.tanh(cell_cur) * gate_output

        return hidden_cur, (hidden_cur, cell_cur)


class NNUpsample4(nn.Module):

    def __init__(self):
        super(NNUpsample4, self ).__init__()
        self.nnup1 = nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None)
        self.nnup2 = nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None)

    def forward(self, x):
        x1 = self.nnup2(self.nnup1(x))
        return x1


class DirectionHeaderRNN(nn.Module):

    def __init__(self):
        super(DirectionHeaderRNN, self ).__init__()
        self.nnupsampler = NNUpsample4()
        self.convrnn = ConvRNNCell(input_c=5, hidden_c=32, kernel_size=3)
        self.conv1 = conv(in_planes=32, out_planes=16, kernel_size=3, stride=1)
        self.insnorm = nn.InstanceNorm2d(16)
        self.res1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res3 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res4 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        self.globalmaxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.linear1_1 = nn.Linear(32,16)
        self.relu = nn.ReLU(inplace=True)
        self.linear1_2 = nn.Linear(16,2)

        self.linear2_1 = nn.Linear(32,16)
        self.linear2_2 = nn.Linear(16,4)

        self.net_rnnout = nn.Sequential(self.conv1, self.insnorm, self.relu, self.res1, self.res2, self.res3, self.res4)

    def forward(self, x_cur, hidden_prev):
        x_up = self.nnupsampler(x_cur)
        hidden_cur = self.convrnn(x_up, hidden_prev)

        x2 = self.net_rnnout(hidden_cur)

        x3 = self.globalmaxpool(x2)
        x3 = x3.squeeze(dim=3)
        x3 = x3.squeeze(dim=2)

        x4 = self.relu(self.linear1_1(x3))
        out_1 = self.linear1_2(x4)
        out1_norm = out_1.norm(dim=1).view(x_cur.shape[0], -1)
        out1 = out_1/out1_norm

        x5 = self.relu(self.linear2_1(x3))
        out_2 = self.linear2_2(x5)

        out2_1 = out_2[:, 0:2]
        out2_2 = out_2[:, 2:4]
        # unit vector norm
        out2_1_norm = out2_1.norm(dim=1).view(x_cur.shape[0], -1)
        out2_1 = out2_1/out2_1_norm

        out2_2_norm = out2_2.norm(dim=1).view(x_cur.shape[0], -1)
        out2_2 = out2_2/out2_2_norm
        return out1, [out2_1, out2_2], hidden_cur


class DirectionHeaderLSTM(nn.Module):

    def __init__(self):
        super(DirectionHeaderLSTM, self ).__init__()
        self.nnupsampler = NNUpsample4()
        self.convrnn = ConvLSTMCell(input_c=5, hidden_c=32, kernel_size=3)
        self.conv1 = conv(in_planes=32, out_planes=16, kernel_size=3, stride=1)
        self.res1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res3 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res4 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        self.globalmaxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.linear1_1 = nn.Linear(32,16)
        self.relu = nn.ReLU(inplace=True)
        self.linear1_2 = nn.Linear(16,2)

        self.linear2_1 = nn.Linear(32,16)
        self.linear2_2 = nn.Linear(16,4)

        self.net_rnnout = nn.Sequential(self.conv1, self.res1, self.res2, self.res3, self.res4)

    def forward(self, x_cur, state_prev):
        x_up = self.nnupsampler(x_cur)
        hidden_cur, state_cur = self.convrnn(x_up, state_prev)

        x2 = self.net_rnnout(hidden_cur)

        x3 = self.globalmaxpool(x2)
        x3 = x3.squeeze(dim=3)
        x3 = x3.squeeze(dim=2)

        x4 = self.relu(self.linear1_1(x3))
        out_1 = self.linear1_2(x4)
        out1_norm = out_1.norm(dim=1).view(x_cur.shape[0], -1)
        out1 = out_1/out1_norm

        x5 = self.relu(self.linear2_1(x3))
        out_2 = self.linear2_2(x5)
        out2_1 = out_2[:, 0:2]
        out2_2 = out_2[:, 2:4]
        # unit vector norm
        out2_1_norm = out2_1.norm(dim=1).view(x_cur.shape[0], -1)
        out2_1 = out2_1/out2_1_norm

        out2_2_norm = out2_2.norm(dim=1).view(x_cur.shape[0], -1)
        out2_2 = out2_2/out2_2_norm
        return out1, [out2_1, out2_2], state_cur



# # no AvgPool
class StateHeaderLSTM(nn.Module):

    def __init__(self):
        super(StateHeaderLSTM, self).__init__()
        self.nnupsampler = NNUpsample4()
        self.convrnn = ConvLSTMCell(input_c=9, hidden_c=32, kernel_size=3)
        self.conv1 = conv(in_planes=32, out_planes=16, kernel_size=3, stride=1)
        self.res1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res3 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res4 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        self.res5 = Residual(kernel_size=3, inplanes=32, planes=32, stride=2)
        self.res6 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        # self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.linear1 = nn.Linear(32, 16)
        self.relu = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(16, 4)

        self.linear_all_1 = nn.Linear(4096, 512)
        self.linear_all_2 = nn.Linear(512, 64)
        self.linear_all_3 = nn.Linear(64, 4)

        self.net_rnnout = nn.Sequential(self.conv1, self.res1, self.res2, self.res3, self.res4, self.res5, self.res6)

    def forward(self, x_cur, state_prev):
        x_up = self.nnupsampler(x_cur)
        hidden_cur, state_cur = self.convrnn(x_up, state_prev)
        x2 = self.net_rnnout(hidden_cur)
        x3 = x2.view(-1, 4096)
        x4 = self.relu(self.linear_all_1(x3))
        x5 = self.relu(self.linear_all_2(x4))
        out = self.relu(self.linear_all_3(x5))

        return out, state_cur


class StateHeaderLSTM_attention(nn.Module):

    def __init__(self):
        super(StateHeaderLSTM_attention, self).__init__()
        self.nnupsampler = NNUpsample4()
        self.convrnn = ConvLSTMCell(input_c=9, hidden_c=32, kernel_size=3)
        self.conv1 = conv(in_planes=32, out_planes=16, kernel_size=3, stride=1)
        self.res1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res3 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res4 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        self.res5 = Residual(kernel_size=3, inplanes=32, planes=32, stride=2)
        self.res6 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        self.attention_conv = conv(in_planes=32, out_planes=1, kernel_size=3, stride=1)
        self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 4)

        self.net_rnnout = nn.Sequential(self.conv1, self.res1, self.res2, self.res3, self.res4, self.res5, self.res6)

    def forward(self, x_cur, state_prev):

        x_up = self.nnupsampler(x_cur)
        hidden_cur, state_cur = self.convrnn(x_up, state_prev)
        x2 = self.net_rnnout(hidden_cur)

        attention = self.attention_conv(x2)
        x3 = x2 * attention
        x3 = self.globalavgpool(x3)
        x3 = x3.squeeze(dim=3)
        x3 = x3.squeeze(dim=2)
        x4 = self.relu(self.linear1(x3))
        out = self.linear2(x4)

        return out, state_cur


class StateHeaderRNN(nn.Module):

    def __init__(self):
        super(StateHeaderRNN, self).__init__()
        self.nnupsampler = NNUpsample4()
        self.convrnn = ConvRNNCell(input_c=9, hidden_c=32, kernel_size=3)
        self.insnorm1 = nn.InstanceNorm2d(32)
        self.conv1 = conv(in_planes=32, out_planes=16, kernel_size=3, stride=1)
        self.insnorm2 = nn.InstanceNorm2d(16)
        self.res1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res3 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res4 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        self.res5 = Residual(kernel_size=3, inplanes=32, planes=32, stride=2)
        self.res6 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        # self.globalavgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.linear1 = nn.Linear(32, 16)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(16, 4)
        # torch.nn.init.kaiming_normal_(self.linear1.weight, a=0, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.linear2.weight, a=0, mode='fan_in', nonlinearity='relu')

        self.net_rnnout = nn.Sequential(self.conv1, self.insnorm2, self.relu, self.res1, self.res2, self.res3,
                                        self.res4)


    def forward(self, x_cur, hidden_prev):
        x_up = self.nnupsampler(x_cur)
        hidden_cur = self.relu(self.insnorm1(self.convrnn(x_up, hidden_prev)))
        x2 = self.net_rnnout(hidden_cur)

        x3 = self.globalavgpool(x2)
        x3 = x3.squeeze(dim=3)
        x3 = x3.squeeze(dim=2)
        x4 = self.relu(self.linear1(x3))
        out = self.linear2(x4)

        return out, hidden_cur


class StateHeaderRNN_attention(nn.Module):

    def __init__(self):
        super(StateHeaderRNN_attention, self).__init__()
        self.nnupsampler = NNUpsample4()
        self.convrnn = ConvRNNCell(input_c=9, hidden_c=32, kernel_size=3)
        self.insnorm1 = nn.InstanceNorm2d(32)
        self.conv1 = conv(in_planes=32, out_planes=16, kernel_size=3, stride=1)
        self.insnorm2 = nn.InstanceNorm2d(16)
        self.res1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res3 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res4 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        # self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.attention_conv_1 = conv(in_planes=32, out_planes=64, kernel_size=3, stride=1)
        self.insnorm_attention_1 = nn.InstanceNorm2d(64)
        self.attention_conv_2 = conv(in_planes=64, out_planes=16, kernel_size=3, stride=1)
        self.insnorm_attention_2 = nn.InstanceNorm2d(16)
        self.attention_conv_3 = conv(in_planes=16, out_planes=1, kernel_size=3, stride=1)
        # self.insnorm_attention_3 = nn.InstanceNorm2d(1)
        self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 4)

        # self.linear_all_1 = nn.Linear(4096, 512)
        # self.linear_all_2 = nn.Linear(512, 64)
        # self.linear_all_3 = nn.Linear(64, 4)

        self.net_rnnout = nn.Sequential(self.conv1, self.insnorm2, self.relu, self.res1, self.res2, self.res3, self.res4)
        self.attention_net = nn.Sequential(self.attention_conv_1, self.insnorm_attention_1, self.relu,
                                           self.attention_conv_2,
                                           self.insnorm_attention_2, self.relu, self.attention_conv_3)

    def forward(self, x_cur, hidden_prev):

        x_up = self.nnupsampler(x_cur)
        hidden_cur = self.relu(self.insnorm1(self.convrnn(x_up, hidden_prev)))
        x2 = self.net_rnnout(hidden_cur)

        attention = self.attention_net(x2)

        x3 = x2 * attention
        x3 = self.globalavgpool(x3)
        x3 = x3.squeeze(dim=3)
        x3 = x3.squeeze(dim=2)
        x4 = self.relu(self.linear1(x3))
        out = self.linear2(x4)
        return out, hidden_cur


# no RNN
class StateHeaderCNN(nn.Module):

    def __init__(self):
        super(StateHeaderCNN, self).__init__()
        self.nnupsampler = NNUpsample4()
        self.conv0 = conv(in_planes=9, out_planes=32, kernel_size=3, stride=1)
        self.insnorm1 = nn.InstanceNorm2d(32)
        self.conv1 = conv(in_planes=32, out_planes=16, kernel_size=3, stride=1)
        self.insnorm2 = nn.InstanceNorm2d(16)
        self.res1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res3 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res4 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        # self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.globalavgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.linear1 = nn.Linear(32,16)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(16,4)
        # torch.nn.init.kaiming_normal_(self.linear1.weight, a=0, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.linear2.weight, a=0, mode='fan_in', nonlinearity='relu')

        self.net_rnnout = nn.Sequential(self.conv1, self.insnorm2, self.relu, self.res1, self.res2, self.res3, self.res4)

    def forward(self, x_cur):
        x_up = self.nnupsampler(x_cur)
        hidden_cur = self.relu(self.insnorm1(self.conv0(x_up)))
        x2 = self.net_rnnout(hidden_cur)
        x3 = self.globalavgpool(x2)
        x3 = x3.squeeze(dim=3)
        x3 = x3.squeeze(dim=2)
        x4 = self.relu(self.linear1(x3))
        out = self.linear2(x4)

        return out

class StateHeaderCNN_attention(nn.Module):

    def __init__(self):
        super(StateHeaderCNN_attention, self).__init__()
        self.nnupsampler = NNUpsample4()
        self.conv0 = conv(in_planes=9, out_planes=32, kernel_size=3, stride=1)
        self.insnorm1 = nn.InstanceNorm2d(32)
        self.conv1 = conv(in_planes=32, out_planes=16, kernel_size=3, stride=1)
        self.insnorm2 = nn.InstanceNorm2d(16)
        self.res1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res3 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res4 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        self.attention_conv_1 = conv(in_planes=32, out_planes=64, kernel_size=3, stride=1)
        self.insnorm_attention_1 = nn.InstanceNorm2d(64)
        self.attention_conv_2 = conv(in_planes=64, out_planes=16, kernel_size=3, stride=1)
        self.insnorm_attention_2 = nn.InstanceNorm2d(16)
        self.attention_conv_3 = conv(in_planes=16, out_planes=1, kernel_size=3, stride=1)
        # self.insnorm_attention_3 = nn.InstanceNorm2d(1)
        # self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.globalavgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.linear1 = nn.Linear(32,16)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(16,4)
        # torch.nn.init.kaiming_normal_(self.linear1.weight, a=0, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.linear2.weight, a=0, mode='fan_in', nonlinearity='relu')

        self.net_rnnout = nn.Sequential(self.conv1, self.insnorm2, self.relu, self.res1, self.res2, self.res3, self.res4)

        self.attention_net = nn.Sequential(self.attention_conv_1, self.insnorm_attention_1, self.relu, self.attention_conv_2,
                                           self.insnorm_attention_2, self.relu, self.attention_conv_3)

    def forward(self, x_cur):

        x_up = self.nnupsampler(x_cur)
        hidden_cur = self.relu(self.insnorm1(self.conv0(x_up)))
        x2 = self.net_rnnout(hidden_cur)

        attention = self.attention_net(x2)

        x3 = x2 * attention

        x3 = self.globalavgpool(x3)
        x3 = x3.squeeze(dim=3)
        x3 = x3.squeeze(dim=2)
        x4 = self.relu(self.linear1(x3))
        out = self.linear2(x4)
        return out


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, device, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel
        self.device=device
        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)#change to NCHW
        else:
            feature = feature.transpose(2, 3).contiguous() #NCWH
            feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature, dim=-1)
        self.pos_x=self.pos_x.to(self.device)
        self.pos_y= self.pos_y.to(self.device)
        softmax_attention=softmax_attention.to(self.device)
        expected_x = torch.sum(self.pos_x* softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_y, expected_x], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


class PositionHeaderLSTM(nn.Module):

    def __init__(self):
        super(PositionHeaderLSTM, self ).__init__()
        self.nnupsampler = NNUpsample4()
        self.convrnn = ConvLSTMCell(input_c=5, hidden_c=32, kernel_size=3)
        self.conv1 = conv(in_planes=32, out_planes=16, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res1_1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res1_2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)

        self.res2_1 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res2_2 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        self.res3_1 = Residual(kernel_size=3, inplanes=32, planes=64, stride=2)
        self.res3_2 = Residual(kernel_size=3, inplanes=64, planes=64, stride=1)

        self.res4_1 = Residual(kernel_size=3, inplanes=64, planes=128, stride=2)
        self.res4_2 = Residual(kernel_size=3, inplanes=128, planes=128, stride=1)

        self.irc1_1 = IRC(kernel_size=1, inplanes=128, planes=64, stride=1)
        self.iruc1 = IRUC(kernel_size=1, inplanes=64, planes=64, stride=1)
        self.irc1_2 = IRC(kernel_size=1, inplanes=64, planes=64, stride=1)

        self.irc2_1 = IRC(kernel_size=1, inplanes=64, planes=32, stride=1)
        self.iruc2 = IRUC(kernel_size=1, inplanes=32, planes=32, stride=1)
        self.irc2_2 = IRC(kernel_size=1, inplanes=32, planes=32, stride=1)

        self.irc3_1 = IRC(kernel_size=1, inplanes=32, planes=16, stride=1)
        self.iruc3 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.irc3_2 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)

        self.irc4_1 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.iruc4 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.irc4_2 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)

        self.iruc5 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.res5 = Residual(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.conv5 = conv(in_planes=16, out_planes=1, kernel_size=1, stride=1)

    def forward(self, x_cur, state_prev):
        x_up = self.nnupsampler(x_cur)
        hidden_cur, state_cur = self.convrnn(x_up, state_prev)

        x0 = self.maxpool(self.conv1(hidden_cur))
        x1 = self.res1_2(self.res1_1(x0))
        x2 = self.res2_2(self.res2_1(x1))
        x3 = self.res3_2(self.res3_1(x2))
        x4 = self.res4_2(self.res4_1(x3))

        y3 = self.irc1_2(self.iruc1(self.irc1_1(x4)))
        y3 += x3
        y2 = self.irc2_2(self.iruc2(self.irc2_1(y3)))
        y2 += x2
        y1 = self.irc3_2(self.iruc3(self.irc3_1(y2)))
        y1 += x1
        y0 = self.irc4_2(self.iruc4(self.irc4_1(y1)))

        out = self.conv5(self.res5(self.iruc5(y0)))

        return out, state_cur

class PositionHeaderRNN(nn.Module):

    def __init__(self):
        super(PositionHeaderRNN, self ).__init__()
        self.nnupsampler = NNUpsample4()
        self.convrnn = ConvRNNCell(input_c=5, hidden_c=32, kernel_size=3)
        self.conv1 = conv(in_planes=32, out_planes=16, kernel_size=7, stride=2)
        self.insnorm = nn.InstanceNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res1_1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res1_2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)

        self.res2_1 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res2_2 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        self.res3_1 = Residual(kernel_size=3, inplanes=32, planes=64, stride=2)
        self.res3_2 = Residual(kernel_size=3, inplanes=64, planes=64, stride=1)

        # self.res4_1 = Residual(kernel_size=3, inplanes=64, planes=128, stride=2)
        self.res4_1 = Residual(kernel_size=3, inplanes=64, planes=128, stride=1)
        self.res4_2 = Residual(kernel_size=3, inplanes=128, planes=128, stride=1)

        self.irc1_1 = IRC(kernel_size=1, inplanes=128, planes=64, stride=1)
        # self.iruc1 = IRUC(kernel_size=1, inplanes=64, planes=64, stride=1)
        self.iruc1 = IRC(kernel_size=1, inplanes=64, planes=64, stride=1)
        self.irc1_2 = IRC(kernel_size=1, inplanes=64, planes=64, stride=1)

        self.irc2_1 = IRC(kernel_size=1, inplanes=64, planes=32, stride=1)
        self.iruc2 = IRUC(kernel_size=1, inplanes=32, planes=32, stride=1)
        self.irc2_2 = IRC(kernel_size=1, inplanes=32, planes=32, stride=1)

        self.irc3_1 = IRC(kernel_size=1, inplanes=32, planes=16, stride=1)
        self.iruc3 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.irc3_2 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)

        self.irc4_1 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.iruc4 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.irc4_2 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)

        self.iruc5 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.res5 = Residual(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.conv5 = conv(in_planes=16, out_planes=1, kernel_size=1, stride=1)

    def forward(self, x_cur, hidden_prev):
        x_up = self.nnupsampler(x_cur)
        hidden_cur = self.convrnn(x_up, hidden_prev)

        x0 = self.maxpool(self.relu(self.insnorm(self.conv1(hidden_cur))))
        x1 = self.res1_2(self.res1_1(x0))
        x2 = self.res2_2(self.res2_1(x1))
        x3 = self.res3_2(self.res3_1(x2))
        x4 = self.res4_2(self.res4_1(x3))

        y3 = self.irc1_2(self.iruc1(self.irc1_1(x4)))
        y3 += x3
        y2 = self.irc2_2(self.iruc2(self.irc2_1(y3)))
        y2 += x2
        y1 = self.irc3_2(self.iruc3(self.irc3_1(y2)))
        y1 += x1
        y0 = self.irc4_2(self.iruc4(self.irc4_1(y1)))

        out = self.conv5(self.res5(self.iruc5(y0)))

        return out, hidden_cur



class PositionHeader_down4(nn.Module):

    def __init__(self):
        super(PositionHeader_down4, self ).__init__()
        self.nnupsampler = NNUpsample4()
        self.convrnn = ConvRNNCell(input_c=12, hidden_c=32, kernel_size=3)
        self.conv1 = conv(in_planes=32, out_planes=16, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res1_1 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)
        self.res1_2 = Residual(kernel_size=3, inplanes=16, planes=16, stride=1)

        self.res2_1 = Residual(kernel_size=3, inplanes=16, planes=32, stride=2)
        self.res2_2 = Residual(kernel_size=3, inplanes=32, planes=32, stride=1)

        self.res3_1 = Residual(kernel_size=3, inplanes=32, planes=64, stride=2)
        self.res3_2 = Residual(kernel_size=3, inplanes=64, planes=64, stride=1)

        self.res4_1 = Residual(kernel_size=3, inplanes=64, planes=128, stride=2)
        self.res4_2 = Residual(kernel_size=3, inplanes=128, planes=128, stride=1)

        self.irc1_1 = IRC(kernel_size=1, inplanes=128, planes=64, stride=1)
        self.iruc1 = IRUC(kernel_size=1, inplanes=64, planes=64, stride=1)
        self.irc1_2 = IRC(kernel_size=1, inplanes=64, planes=64, stride=1)

        self.irc2_1 = IRC(kernel_size=1, inplanes=64, planes=32, stride=1)
        self.iruc2 = IRUC(kernel_size=1, inplanes=32, planes=32, stride=1)
        self.irc2_2 = IRC(kernel_size=1, inplanes=32, planes=32, stride=1)

        self.irc3_1 = IRC(kernel_size=1, inplanes=32, planes=16, stride=1)
        self.iruc3 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.irc3_2 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)

        self.irc4_1 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.iruc4 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.irc4_2 = IRC(kernel_size=1, inplanes=16, planes=16, stride=1)

        self.iruc5 = IRUC(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.res5 = Residual(kernel_size=1, inplanes=16, planes=16, stride=1)
        self.conv5 = conv(in_planes=16, out_planes=1, kernel_size=1, stride=1)

    def forward(self, x_cur, hidden_prev):
        x_up = self.nnupsampler(x_cur)
        hidden_cur = self.convrnn(x_up, hidden_prev)
        x0 = self.maxpool(self.conv1(hidden_cur))
        x1 = self.res1_2(self.res1_1(x0))
        x2 = self.res2_2(self.res2_1(x1))
        x3 = self.res3_2(self.res3_1(x2))

        y2 = self.irc2_2(self.iruc2(self.irc2_1(x3)))
        y2 += x2
        y1 = self.irc3_2(self.iruc3(self.irc3_1(y2)))
        y1 += x1
        y0 = self.irc4_2(self.iruc4(self.irc4_1(y1)))

        out = self.conv5(self.res5(self.iruc5(y0)))

        return out, hidden_cur




class TopologyRNN(nn.Module):

    def __init__(self,state_net="CNN", pos_net="RNN", direction_net="RNN"):
        super(TopologyRNN, self).__init__()
        self.globalTeatureNet = GlobalFeature()
        self.DTNet = DistanceTransform()
        if direction_net == "RNN":
            self.directionHeader = DirectionHeaderRNN()
        elif direction_net == "LSTM":
            self.directionHeader = DirectionHeaderLSTM()

        if state_net == "RNN":
            '''Use StateHeader_RNN'''
            self.stateHeader = StateHeaderRNN()
            # self.stateHeader = StateHeaderRNN_attention()
        elif state_net == "LSTM":
            # self.stateHeader = StateHeaderLSTM()
            self.stateHeader = StateHeaderLSTM_attention()
        elif state_net == "CNN":
            self.stateHeader = StateHeaderCNN()
        elif state_net == "CNN_A":
            '''Use StateHeaderCNN_Attention'''
            self.stateHeader = StateHeaderCNN_attention()

        if pos_net == "RNN":
            self.positionHeader = PositionHeaderRNN()
        elif pos_net == "LSTM":
            self.positionHeader = PositionHeaderLSTM()

    def forward(self, x):
        globalFeature = self.globalTeatureNet(x)
        DT = self.DTNet(globalFeature)
        concatFeature = torch.cat((globalFeature, DT), 1)

        return DT, concatFeature


