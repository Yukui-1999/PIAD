import torch.nn as nn
import torch.nn.functional as F
import torch
import math
# adapt from https://github.com/MIC-DKFZ/BraTS2017
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
def exists(x):
    return x is not None
def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        step = step.view(1, -1).expand(noise_level.shape[0], -1)  # Expanding step to match the noise_level's shape
        encoding = noise_level * torch.exp(-math.log(1e4) * step)
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding



class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.noise_func = FeatureWiseAffine(3, in_channels)
        
        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x,t):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)

        x1 = self.noise_func(x1, t)

        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureWiseAffine, self).__init__()
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1, 1)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels=4, base_channels=16, num_classes=4):
        super(Unet, self).__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)

        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

        self.noise_level_mlp = nn.Sequential(
                    PositionalEncoding(3),
                    nn.Linear(6, 3 * 4),
                    Swish(),
                    nn.Linear(3 * 4, 3)
                )
    
    def forward(self, x,cfd):
        t = self.noise_level_mlp(cfd) 
        print(f"t:{t.shape}")
        x = self.InitConv(x)       # (1, 16, 128, 128, 128)
        
        x1_1 = self.EnBlock1(x,t)
        x1_2 = self.EnDown1(x1_1)  # (1, 32, 64, 64, 64)

        x2_1 = self.EnBlock2_1(x1_2,t)
        x2_1 = self.EnBlock2_2(x2_1,t)
        x2_2 = self.EnDown2(x2_1)  # (1, 64, 32, 32, 32)

        x3_1 = self.EnBlock3_1(x2_2,t)
        x3_1 = self.EnBlock3_2(x3_1,t)
        x3_2 = self.EnDown3(x3_1)  # (1, 128, 16, 16, 16)

        x4_1 = self.EnBlock4_1(x3_2,t)
        x4_2 = self.EnBlock4_2(x4_1,t)
        x4_3 = self.EnBlock4_3(x4_2,t)
        output = self.EnBlock4_4(x4_3,t)  # (1, 128, 16, 16, 16)

        return x1_1,x2_1,x3_1,output


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 1, 64, 128, 128), device=cuda0)
        cfd = torch.rand((1, 3), device=cuda0)
        # model = Unet1(in_channels=4, base_channels=16, num_classes=4)
        model = Unet(in_channels=1, base_channels=16, num_classes=4)
        model.cuda()
        output = model(x,cfd)
        print('x1_1:', output[0].shape)
        print('x2_1:', output[1].shape)
        print('x3_1:', output[2].shape)
        print('output:', output[3].shape)
