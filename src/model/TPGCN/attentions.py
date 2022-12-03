import torch
from torch import nn

class ST_Part_Att(nn.Module):
    def __init__(self, channel, parts, reduct_ratio, bias, **kwargs):
        super(ST_Part_Att, self).__init__()

        self.parts = parts
        self.joints = nn.Parameter(self.get_corr_joints(), requires_grad=False)
        self.mat = nn.Parameter(self.get_mean_matrix(), requires_grad=False)
        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
    
        self.bn = nn.BatchNorm2d(channel)
        self.act = Swish(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        P = len(self.parts)
        res = x

        x_t = x.mean(3, keepdims=True) # N,C,T,1
        # x_v = x.mean(2, keepdims=True).transpose(2, 3) # N,C,V,1
        # x_p = torch.einsum('nclv,vp->nclp',(x.mean(2, keepdims=True), self.mat)).transpose(2, 3) # N,C,P,1
        x_p = (x.mean(2, keepdims=True) @ self.mat).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_p], dim=2)) # N,C,(T+P),1
        x_t, x_p = torch.split(x_att, [T, P], dim=2) 
        x_t_att = self.conv_t(x_t).sigmoid() # N,C,T,1
        
        x_p_att = self.conv_v(x_p.transpose(2, 3)).sigmoid() # N,C,1,P
        x_v_att = x_p_att.index_select(3, self.joints) # N,C,1,V

        x_att = x_t_att * x_v_att
        return self.act(self.bn(x * x_att) + res)
    
    def get_corr_joints(self):
        num_joints = sum([len(part) for part in self.parts])
        joints = [j for i in range(num_joints) for j in range(len(self.parts)) if i in self.parts[j]]
        return torch.LongTensor(joints)
    
    def get_mean_matrix(self):
        num_joints = sum([len(part) for part in self.parts])
        Q = torch.zeros(num_joints, len(self.parts))
        for j in range(len(self.parts)):
            n = len(self.parts[j])
            for joint in self.parts[j]:
                Q[joint][j] = 1.0/n
        return Q

class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

        self.bn = nn.BatchNorm2d(channel)
        self.act = Swish(inplace=True)

    def forward(self, x):
        res = x
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return self.act(self.bn(x * x_att) + res)

class Part_Att(nn.Module):
    def __init__(self, channel, parts, **kwargs):
        super(Part_Att, self).__init__()

        self.parts = parts
        self.joints = get_corr_joints(parts)

        inter_channel = channel // 4

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channel*len(self.parts), kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x

        x_att = self.softmax(self.fcn(x).view(N, C, len(self.parts)))
        x_att = torch.split(x_att, 1, dim=-1)
        x_att = [x_att[self.joints[i]].expand_as(x[:,:,:,i]) for i in range(V)]
        x_att = torch.stack(x_att, dim=-1)
        return self.relu(self.bn(x * x_att) + res)

def get_corr_joints(parts):
    num_joints = max([max(part) for part in parts]) + 1
    res = []
    for i in range(num_joints):
        for j in range(len(parts)):
            if i in parts[j]:
                res.append(j)
                break
    return torch.Tensor(res).long()

class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())