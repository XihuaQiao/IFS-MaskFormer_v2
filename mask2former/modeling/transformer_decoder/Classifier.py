import torch
from torch import nn
import torch.nn.functional as F

class CoMFormerIncClassifier(nn.Module):
    def __init__(self, classes, norm_feat=False, channels=256, bias=True):
        super().__init__()
        self.cls = nn.ModuleList(
            [nn.Linear(channels, c, bias=bias) for c in classes])
        self.norm_feat = norm_feat

    def forward(self, x):
        if self.norm_feat:
            x = F.normalize(x, p=2, dim=3)
        out = []
        for mod in self.cls:
            out.append(mod(x))
        # out.append(self.cls[0](x))  # put as last the void class
        return torch.cat(out, dim=2)
    

class CoMFormerCosClassifier(nn.Module):
    def __init__(self, classes, norm_feat=True, channels=256, scaler=10.):
        super().__init__()
        self.cls = nn.ModuleList(
            [nn.Linear(channels, c, bias=False) for c in classes])
        self.norm_feat = norm_feat
        self.scaler = scaler

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = []
        for mod in self.cls[1:]:
            out.append(self.scaler * F.linear(x, F.normalize(mod.weight, dim=1, p=2)))
        out.append(self.scaler * F.linear(x, F.normalize(self.cls[0].weight, dim=1, p=2)))  # put as last the void class
        return torch.cat(out, dim=2)


class MicroSegHead(nn.Module):
    def __init__(self, classes, hidden_dim):
        super().__init__()
        self.proposal_head = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, c, 1, bias=True)
            ) for c in classes]
        )
        
        self._init_weight()

    def forward(self, features):
        # feature - B,Q,D

        B_, N_, C_ = features.shape
        features = features.view(B_ * N_, -1)         # B*Q,D
        features = features.unsqueeze(-1).unsqueeze(-1)

        cl = [ph(features) for ph in self.proposal_head]

        cl = torch.cat(cl, dim=1)
        cl = cl.view(B_, N_, cl.shape[1])   # B,Q,C+1

        return cl

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MLPHead(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
