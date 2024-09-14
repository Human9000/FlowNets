import torch

from torch import nn
from torch.nn import functional as F


class Shuffle1d(nn.Module):
    def __init__(self, groups=16):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, channels, length = x.shape
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, length)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, channels, length)
        return x


class SortAttn(nn.Module):
    def __init__(self, channels,  groups=8):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, groups=groups),
            Shuffle1d(groups),
            nn.Conv1d(channels, channels, 1, groups=groups),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.shape[-1] < 1:
            return x
        # 对隐藏层进行通道排序
        sorted_x, indices = torch.sort(x, dim=1, stable=True)
        attn = self.attn(torch.cat((sorted_x, x), dim=1))
        return x * attn


# 流处理的conv1d层，可以处理长度为l的向量，也可以向之前长度为l的向量中添加一个长度为1的向量
class FSSConv1dCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Sequential(
             nn.Conv1d(in_channels, out_channels, 3, padding=0, stride=1, groups=groups),
             Shuffle1d(groups=groups),
             nn.Conv1d(out_channels, out_channels, 3, padding=0, stride=1, groups=groups),
             Shuffle1d(groups=groups),
             SortAttn(out_channels, groups),
             nn.Conv1d(out_channels, out_channels, kernel_size-4, padding=0, stride=stride, groups=groups),
        )
        self.k = kernel_size
        self.s = stride
        self.out_channels = out_channels 
        self.in_channels = in_channels 
        self.init_state()

    def init_state(self): 
        self.in_state = torch.zeros(1, self.in_channels, 0)
        self.out_state = torch.zeros(1, self.out_channels, 1)

    def forward(self, x):  # x: [batch_size, in_channels, length]  
        x = torch.cat((self.in_state.to(x.device), x), dim=2)

        # print(x.shape)
        if x.shape[-1] < self.k:
            self.in_state = x
            self.h = x.shape[-1]
            return torch.zeros(x.shape[0], self.out_channels, 0).to(x.device)

        y = self.conv(x)

        # 记录最后一次有效输出
        self.out_state = y[..., ]

        # 计算状态转移向量的长度
        l = x.shape[2]
        k = self.k
        s = self.s
        h = k-s + (l-k) % s
        # 记录状态转移向量 
        self.in_state = x[:, :, -h:]
        return y

# flow-sort-shuffle-conv1d
class FSSConv1d(nn.Module):
    def __init__(self,
                 in_channel=12,
                 seg_number=4,
                 cls_number=4,
                 hidden=128,
                 kernel=7,
                 num_layers=5,
                 groups=16):
        super().__init__()
        self.layer = num_layers
        self.kernel = kernel 
        
        self.backbone = nn.Sequential(
            FSSConv1dCell(in_channel, hidden, kernel, 2, 1), 
            nn.ReLU(),
            *[
                nn.Sequential(
                    FSSConv1dCell(hidden, hidden, kernel, 2, groups), 
                    nn.ReLU(),
                ) for _ in range(num_layers-2)
            ],
            FSSConv1dCell(hidden, hidden, kernel, 2, 1),
        )
        
        # 计算全部特征层的通道数
        dechannel = num_layers * hidden

        # 分类输出头
        self.cls_head = nn.Sequential(
            nn.Conv1d(dechannel, dechannel, 1, groups=groups*num_layers),
            SortAttn(dechannel),
            Shuffle1d(groups),
            nn.ReLU(),
            nn.Conv1d(dechannel, hidden, 1, groups=groups),
            SortAttn(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, cls_number, 1),
        )

        # 分割输出头
        self.seg_head = nn.Sequential(
            nn.Conv1d(dechannel, dechannel, 3, 1, 1, groups=groups*num_layers),
            SortAttn(dechannel),
            Shuffle1d(groups),
            nn.ReLU(),
            nn.Conv1d(dechannel, hidden, 3, 1, 1, groups=groups),
            SortAttn(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, seg_number, 3, 1, 1),
        )
        # self.init_state()

    def get_view(self):
        view = self.kernel
        for i in range(1, self.layer):
            view += (self.kernel - 1) * 2**i
        return view

    def get_stride(self):
        return 2**self.layer

    def init_state(self):
        for module in self.modules():
            if isinstance(module, FSSConv1dCell):
                module.init_state()
    
    # 提取分类特征
    def get_cls_features(self):
        last_effective_outs = []
        for module in self.backbone.modules():
            if isinstance(module, FSSConv1dCell):
                f = module.out_state[..., -1:]
                f = f.to(self.cls_head[0].weight.device)
                last_effective_outs.append(f)
        last_effective_outs = torch.cat(last_effective_outs, dim=1)
        return last_effective_outs
    
    # 提取分割特征
    def get_seg_features(self, size):
        last_effective_outs = []
        for module in self.backbone.modules():
            if isinstance(module, FSSConv1dCell):
                f = module.out_state
                f = f.to(self.cls_head[0].weight.device)
                f = F.pad(f , (size[-1] - f.shape[-1], 0))
                last_effective_outs.append(f)
        last_effective_outs = torch.cat(last_effective_outs, dim=1)
        return last_effective_outs

    def forward(self, x):
        self.backbone(x)  # 提取特征
        # 分类
        cls = self.cls_head(self.get_cls_features())
        # 分割
        seg = self.seg_head(self.get_seg_features(x.shape[-1:]))
        return seg, cls


def test_convflow():
    conv = FSSConv1dCell(1, 1, 5, 2)
    x = torch.randn(1, 1, 100)
    # 并行卷积
    y, hidden = conv(x)
    print(y)
    # 流式卷积，增量添加数据
    y, hidden = conv(x[..., :6])
    s = conv.s * 2
    for i in range(6, x.shape[-1], s):
        yi, hidden = conv(x[..., i:i+s], hidden)
        y = torch.cat((y, yi), dim=2)
    print(y)


def test_flowecg():
    X = torch.randn(1, 12, 5000).cuda()
    # 并行卷积
    flow = FSSConv1d(hidden=128, kernel=13, num_layers=8).cuda()

    print(flow.get_view(), flow.get_stride())
    # print(flow)
    seg1, cls1 = flow(X)
    print(seg1, cls1)

    # 流式卷积，增量添加数据
    seg2 = []
    cls2 = None
    step = 100
    flow.init_state()  # 初始化状态转移向量
    for i in range(0, X.shape[-1], step):  # 后续每次输入的长度要满足步长
        seg, cls2 = flow(X[..., i:i+step])
        seg2.append(seg)
    seg2 = torch.cat(seg2, dim=-1)
    print(seg2, cls2)


def test_view(k, s, layer=5):
    views = [k]
    for i in range(1, layer):
        view = views[-1] + (k - 1) * s**i
        views.append(view)
    print("views", views)
    print("steps", [s**l for l in range(1, layer+1)])


if __name__ == '__main__':
    # test_convflow()
    # test_view(7, 2, layer=3)
    test_flowecg()
