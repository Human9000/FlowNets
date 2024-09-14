from typing import List
import torch
import torch.nn as nn 

class Shuffle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape = x.shape
        x = x.transpose(-1, -2).contiguous().view(shape).contiguous()
        return x


class SortAttn(nn.Module):
    def __init__(self, hidden_size, group=4):
        super(SortAttn, self).__init__()
        group_hidden = hidden_size * 2 // group
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Flatten(1), nn.Unflatten(1, (group, -1)),  # 重新分组
            Shuffle(),  # shuffle
            nn.Linear(group_hidden, group_hidden, bias=False),
            Shuffle(),  # shuffle
            nn.Linear(group_hidden, group_hidden, bias=False),
            nn.ReLU(),  # 非线性激活
            nn.Flatten(1),  # 合组
            nn.Linear(group_hidden * group, hidden_size, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 对隐藏层进行通道排序
        sorted_x, indices = torch.sort(x, dim=1, stable=True)
        # 残差链接排序通道
        se_hidden = torch.stack((x, sorted_x), dim=1)  # b, 2, d
        # 生成注意力分数
        attention_scores = self.attention(se_hidden)  # b,d
        # 叠加注意力结果
        return x * attention_scores


class FSSGruCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FSSGruCell, self).__init__()
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.attn = SortAttn(hidden_size)

    def forward(self, x, hidden:torch.Tensor):
        new_hidden = self.gru_cell(x, hidden)  # b,d
        new_hidden = self.attn(new_hidden)  # b,d
        return new_hidden


class FSSGruBackBone(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FSSGruBackBone, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru_cells = nn.ModuleList([FSSGruCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.hidden_0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size), requires_grad=True)

    def forward(self, x, hidden=torch.zeros(1)):
        batch_size, seq_len, _ = x.size() 
        hiddens:List[torch.Tensor]  = []
        if hidden.ndim == 1:  # 初始化隐藏层
            hidden = self.hidden_0.to(x.device) *\
                torch.ones(1, batch_size, 1).to(x.device)
        hiddens.append(hidden) 
        outputs = []

        for t in range(seq_len):
            inp = x[:, t]
            hidden_t:List[torch.Tensor] = [torch.zeros(0) for _ in range( self.num_layers) ]
            for layer, gru_cell in enumerate(self.gru_cells):
                hidden_t[layer] = gru_cell(inp, hiddens[-1][layer])
                inp = hidden_t[layer]
            hidden_t = torch.stack(hidden_t, dim=0)
            hiddens.append(hidden_t)
            outputs.append(inp)
        return torch.stack(outputs, dim=1), hiddens[-1]


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Log(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


# Flow-Sort-Shuffle-Gru-Net
class FSSGruNet(nn.Module):
    def __init__(self, in_channel, window, seg_number=4, cls_number=4, hidden=64, num_layers=8, groups=8):
        super().__init__()
        self.win = window
        
        self.backbone = FSSGruBackBone(input_size=in_channel * window,
                           hidden_size=hidden,
                           num_layers=num_layers, 
                           ) 

        self.seg_head = nn.Sequential(
            nn.Unflatten(2, (groups, -1)),  # 分组
            nn.Linear(hidden // groups, seg_number),
            Shuffle(),
            nn.Linear(seg_number, seg_number, ),
            Shuffle(),
            nn.Linear(seg_number, seg_number, ),
            nn.Flatten(2), nn.Unflatten(2, (seg_number, groups)),  # 重组
            nn.Linear(groups, window,),
            Transpose(-1, -2),
            nn.Flatten(1, 2),
        ) 
        
        self.cls_head = nn.Sequential( 
            nn.Unflatten(1, (groups, -1)),  # 分组 ,b,d -> b,g,d//g
            nn.Linear(hidden // groups, cls_number),
            Shuffle(),  # shuffle
            nn.Linear(cls_number, cls_number,),
            Shuffle(),
            nn.Linear(cls_number, cls_number,),
            nn.ReLU(),
            nn.Flatten(1),  # 合组
            nn.Linear(cls_number * groups, cls_number,),
        )
        

    def forward(self, x, h0=torch.zeros(1), seg0=torch.zeros(1)): 
        b, l, c = x.shape
        x = x.reshape(b, l // self.win, self.win * c) 
        ys, h = self.backbone(x, h0)  # (b,l,w*64) (l,b,64)
        seg = self.seg_head(ys)  # b,l,w,4



        if seg0.ndim == 1:
            seg0 = torch.zeros_like(seg).to(seg.device) 
        
        # 10 s 的 分割延迟，保证边界的连贯
        pool_seg = torch.cat((seg0[:,-10:], seg), dim=1)
        pool_seg = torch.avg_pool1d(pool_seg.transpose(-1,-2), 11, 1).transpose(-1,-2) 

        lw = ys.shape[1]
        cls = self.cls_head(ys.reshape(b*lw,-1))  # b,l,4
        cls = cls.reshape(b,lw,-1).mean(dim=1)
        return (pool_seg, cls), (ys, h, seg)

 

if __name__ == '__main__':
    model = FSSGruNet(in_channel=12,
                         window=100,
                         seg_number=4,
                         cls_number=4,
                         hidden=96,
                         num_layers=16,
                         groups=8,
                         ).cuda()  # 12个通道，每个通道有5个特征
    batch_size = 10 
 