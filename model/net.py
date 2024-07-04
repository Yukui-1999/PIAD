import torch.nn as nn
import torch
from torchsummary import  summary
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn.functional as F
import math
import numpy as np
class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels,n):
        super(UNetGenerator, self).__init__()
        
        self.encoder1 = self.contracting_block(in_channels, n, kernel_size=3, stride=1,padding=1)
        self.encoder2 = self.contracting_block(n, 2*n, kernel_size=3, stride=1,padding=1)
        self.encoder3 = self.contracting_block(2*n, 4*n, kernel_size=3, stride=1,padding=1)
        self.encoder4 = self.contracting_block(4*n, 8*n, kernel_size=3, stride=1,padding=1)
        self.encoder5 = self.contracting_block(8*n, 8*n, kernel_size=3, stride=1,padding=1)


        # 解码器部分
        self.decoder1 = self.expanding_block(8*n, 8*n, kernel_size=4, stride=2,padding=1)
        self.decoder2 = self.expanding_block(16*n, 4*n, kernel_size=4, stride=2,padding=1)
        self.decoder3 = self.expanding_block(8*n, 2*n, kernel_size=4, stride=2,padding=1)
        self.decoder4 = self.expanding_block(4*n, n, kernel_size=4, stride=2,padding=1)


        self.decoder5 = nn.Sequential(
            nn.ConvTranspose3d(2*n, out_channels, kernel_size=4, stride=2,padding=1),
            nn.Sigmoid()
        )

    def contracting_block(self, in_channels, out_channels, **kwargs):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, **kwargs),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(p=0.2),
            nn.MaxPool3d((2,2,2)), 
            # conv 3,1,1 needs
        )
        return block

    def expanding_block(self, in_channels, out_channels, **kwargs):
        block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, **kwargs),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(p=0.2)
        )
        return block


    def forward(self, x):
        # print(f"x.shape:{x.shape}")
        encode1 = self.encoder1(x)
        # print(f"encode1.shape:{encode1.shape}")
        encode2 = self.encoder2(encode1)  
        # print(f"encode2.shape:{encode2.shape}")
        encode3 = self.encoder3(encode2)   
        # print(f"encode3.shape:{encode3.shape}")   
        encode4 = self.encoder4(encode3)     
        # print(f"encode4.shape:{encode4.shape}")
        encode5 = self.encoder5(encode4) 
        # print(f"encode5.shape:{encode5.shape}")

        decode1 = self.decoder1(encode5) 
        # print(f"decode1.shape:{decode1.shape}")
         
        decode1 = torch.cat([decode1, encode4], dim=1)
        
        decode2 = self.decoder2(decode1) 
        # print(f"decode2.shape:{decode2.shape}")
      
        decode2 = torch.cat([decode2, encode3], dim=1)
      
        decode3 = self.decoder3(decode2)   
        # print(f"decode3.shape:{decode3.shape}")
       
        decode3 = torch.cat([decode3, encode2], dim=1)
       
        decode4 = self.decoder4(decode3) 
        # print(f"decode4.shape:{decode4.shape}")
       
        decode4 = torch.cat([decode4, encode1], dim=1)
       
        decode5 = self.decoder5(decode4)
        # print(f"decode5.shape:{decode5.shape}")
      
        return decode5


class Discriminator(nn.Module):
    def __init__(self,in_channels=1,n=8):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2,n , normalization=True),
            *discriminator_block(n, 2*n),
            *discriminator_block(2*n, 4*n),
            *discriminator_block(4*n, 8*n),
            nn.Conv3d(8*n, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
      
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class GANLoss(nn.Module):


    def __init__(self, target_real_label=1.0, target_fake_label=0.0):

        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):

        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)

        return loss

class UNetClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetClassifier, self).__init__()

        self.encoder1 = self.contracting_block(in_channels, 128, kernel_size=4, stride=2,padding=1)
        self.encoder2 = self.contracting_block(128, 256, kernel_size=4, stride=2,padding=1)
        self.encoder3 = self.contracting_block(256, 512, kernel_size=4, stride=2,padding=1)
        self.encoder4 = self.contracting_block(512, 1024, kernel_size=4, stride=2,padding=1)
        self.encoder5 = self.contracting_block(1024, 1024, kernel_size=4, stride=2,padding=1)


        # 解码器部分
        self.decoder1 = self.expanding_block(1024, 1024, kernel_size=4, stride=2,padding=1)
        self.decoder2 = self.expanding_block(2048, 512, kernel_size=4, stride=2,padding=1)
        self.decoder3 = self.expanding_block(1024, 256, kernel_size=4, stride=2,padding=1)
        self.decoder4 = self.expanding_block(512, 128, kernel_size=4, stride=2,padding=1)

        self.decoder5 = nn.Sequential(
            nn.ConvTranspose3d(256, out_channels, kernel_size=4, stride=2,padding=1),
            # PrintShape(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # 全局平均池化
            # PrintShape(),
            nn.Flatten(),  # 把维度展平为一维
            # PrintShape(),
            nn.Linear(2, 1),  # 全连接层
            nn.Sigmoid()  # 转换为类别概率
        )

    def contracting_block(self, in_channels, out_channels, **kwargs):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, **kwargs),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block

    def expanding_block(self, in_channels, out_channels, **kwargs):
        block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, **kwargs),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # print(f"x.shape:{x.shape}")
        encode1 = self.encoder1(x)
        # print(f"encode1.shape:{encode1.shape}")
        encode2 = self.encoder2(encode1)  
        # print(f"encode2.shape:{encode2.shape}")
        encode3 = self.encoder3(encode2)   
        # print(f"encode3.shape:{encode3.shape}")   
        encode4 = self.encoder4(encode3)     
        # print(f"encode4.shape:{encode4.shape}")
        encode5 = self.encoder5(encode4) 
        # print(f"encode5.shape:{encode5.shape}")

        decode1 = self.decoder1(encode5) 
        # print(f"decode1.shape:{decode1.shape}")
       
        decode1 = torch.cat([decode1, encode4], dim=1)
     
        decode2 = self.decoder2(decode1) 
        # print(f"decode2.shape:{decode2.shape}")
     
        decode2 = torch.cat([decode2, encode3], dim=1)
   
        decode3 = self.decoder3(decode2)   
        # print(f"decode3.shape:{decode3.shape}")
    
        decode3 = torch.cat([decode3, encode2], dim=1)
       
        decode4 = self.decoder4(decode3) 
        # print(f"decode4.shape:{decode4.shape}")
      
        decode4 = torch.cat([decode4, encode1], dim=1)
      
        decode5 = self.decoder5(decode4)
        # print(f"decode5.shape:{decode5.shape}")
     
        return decode5
    
class ResNetClassifier(nn.Module):
    def __init__(self,inchannel,outchannel) -> None:
        super().__init__()
        weights = R3D_18_Weights.DEFAULT
        self.model = r3d_18(weights=weights)
        self.linear = nn.Linear(400,outchannel)
        
    def forward(self, x):
        out = self.model(x)
        out = self.linear(out)
        
        return out

class _3D_CNN(nn.Module):

    def __init__(self, num_classes):
        super(_3D_CNN, self).__init__()
        self.conv1 = nn.Conv3d(2, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.BN3d1 = nn.BatchNorm3d(num_features=8)
        self.BN3d2 = nn.BatchNorm3d(num_features=16)
        self.BN3d3 = nn.BatchNorm3d(num_features=32)
        self.BN3d4 = nn.BatchNorm3d(num_features=64)
        self.BN3d5 = nn.BatchNorm3d(num_features=128)
        self.pool1 = nn.AdaptiveMaxPool3d((61, 73, 61))
        self.pool2 = nn.AdaptiveMaxPool3d((31, 37, 31))
        self.pool3 = nn.AdaptiveMaxPool3d((16, 19, 16))
        self.pool4 = nn.AdaptiveMaxPool3d((8, 10, 8))
        self.pool5 = nn.AdaptiveMaxPool3d((4, 5, 4))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(10240, 1300)
        self.fc2 = nn.Linear(1300, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.BN3d1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.BN3d2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.BN3d3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.BN3d4(self.conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.BN3d5(self.conv5(x)))
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
class Small3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Small3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.BN3d1 = nn.BatchNorm3d(num_features=8)
        self.BN3d2 = nn.BatchNorm3d(num_features=16)
        self.BN3d3 = nn.BatchNorm3d(num_features=32)
        self.BN3d4 = nn.BatchNorm3d(num_features=64)
        self.BN3d5 = nn.BatchNorm3d(num_features=128)
        self.BN3d6 = nn.BatchNorm3d(num_features=256)
        self.pool1 = nn.AdaptiveMaxPool3d((48, 96, 96))
        self.pool2 = nn.AdaptiveMaxPool3d((32, 64, 64))
        self.pool3 = nn.AdaptiveMaxPool3d((16, 32, 32))
        self.pool4 = nn.AdaptiveMaxPool3d((8, 16, 16))
        self.pool5 = nn.AdaptiveMaxPool3d((4, 8, 8))
        self.pool6 = nn.AdaptiveMaxPool3d((2, 4, 4))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256*2*4*4, 1000)
        self.fc2 = nn.Linear(1000, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.BN3d1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.BN3d2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.BN3d3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.BN3d4(self.conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.BN3d5(self.conv5(x)))
        x = self.pool5(x)
        x = F.relu(self.BN3d6(self.conv6(x)))
        x = self.pool6(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, dropout=0, norm_groups=32):
        super().__init__()

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
       
        batch, channel, depth, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
     
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, depth, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchdw, bnczyx -> bnhdwzyx", query, key
        ).contiguous() / math.sqrt(channel)
     
        attn = attn.view(batch, n_head, depth, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, depth, height, width, depth, height, width)

        out = torch.einsum("bnhdwzyx, bnczyx -> bnchdw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, depth, height, width))

        return out + input



class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x):
        x = self.res_block(x)
        if(self.with_attn):
            x = self.attn(x)
        return x

class ResEncoder(nn.Module):
    def __init__(
        self,
        in_channel=1,
        inner_channel=16,
        norm_groups=16,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=((4,8,8),),
        res_blocks=1,
        dropout=0.2,
        image_size=(64,128,128)
    ):
        super().__init__()
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv3d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(pre_channel, channel_mult, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = tuple(i//2 for i in now_res)
        self.downs = nn.ModuleList(downs)
        self.mid = ResnetBlocWithAttn(pre_channel, pre_channel, norm_groups=norm_groups, dropout=dropout, with_attn=True)

        self.feat_channels = feat_channels
        self.now_res = now_res


    def forward(self, x):
        feat = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x)
            else:
                x = layer(x)
            feat.append(x)
        x = self.mid(x)
        return x,feat

class ResDecoder(nn.Module):
    def __init__(
        self,
        in_channel=128,
        inner_channel=16,
        out_channel=1,
        norm_groups=16,
        now_res=(4,8,8),
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=((4,8,8),),
        res_blocks=1,
        dropout=0.2,
        feat_channels=[16, 16, 16, 32, 32, 64, 64, 128, 128, 128],
     
    ):
        super().__init__()
        self.mid = ResnetBlocWithAttn(in_channel, in_channel, norm_groups=norm_groups, dropout=dropout, with_attn=True)
        pre_channel=in_channel
        ups = []
        self.now_res=now_res
       
        num_mults = len(channel_mults)
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (self.now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                # print(f"feat_channels[-1]:{feat_channels[-1]}")
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult,  norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                # print(f"pre_channel{pre_channel}")
                ups.append(Upsample(pre_channel))
                self.now_res = self.now_res*2

        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, feat):
            x = self.mid(x)
            for layer in self.ups:
                if isinstance(layer, ResnetBlocWithAttn):
                    # print(f"x.shape:{x.shape},feats.shape:{self.feats[-1].shape}")
                    # print(layer)
                    x = layer(torch.cat((x, feat.pop()), dim=1))
                else:
                    x = layer(x)
            return self.sigmoid(self.final_conv(x))

class ResClassifier(nn.Module):
    def __init__(
        self,
        in_channel=128,
        inner_channel=16,
        out_channel=1,
        norm_groups=16,
        now_res=(4,8,8),
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=((4,8,8),),
        res_blocks=1,
        dropout=0.2,
        
    ):
        super().__init__()
        self.mid = ResnetBlocWithAttn(in_channel, in_channel, norm_groups=norm_groups, dropout=dropout, with_attn=True)
        pre_channel=in_channel
        ups = []
        self.now_res=now_res
        
        num_mults = len(channel_mults)
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (self.now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                # print(f"feat_channels[-1]:{feat_channels[-1]}")
                ups.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult,  norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                # print(f"pre_channel{pre_channel}")
                ups.append(Upsample(pre_channel))
                self.now_res = self.now_res*2

        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(out_channel*64*128*128,out_channel)
        # self.fc2 = nn.Linear(1024,out_channel)

    def forward(self, x):
            x = self.mid(x)
            for layer in self.ups:
                if isinstance(layer, ResnetBlocWithAttn):
                    # print(f"x.shape:{x.shape},feats.shape:{self.feats[-1].shape}")
                    # print(layer)
                    x = layer(x)
                else:
                    x = layer(x)
            x = self.final_conv(x)   
            x = x.view(x.size(0), -1) 
            x = self.fc1(x)
            # x = self.fc2(x)
            return x

import torch.nn as nn
import torch.nn.functional as F
class MADL_classifier(nn.Module):
    def __init__(self,num_classes=1):
        super(MADL_classifier, self).__init__()
        
        # Initial Conv layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # Stride of 2 to reduce dimensions
        
        # Conv layers
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Another stride of 2
        
        # Batch normalization
        self.BN2d1 = nn.GroupNorm(8,16)
        self.BN2d2 = nn.GroupNorm(8,32)
        self.BN2d3 = nn.GroupNorm(8,64)
        
        # Max pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce spatial dimensions
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        
        self.fc1 = nn.Linear(64, num_classes)


    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension: [batch, 1, 2048, 512]

        x = F.relu(self.BN2d1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.BN2d2(self.conv2(x)))
        
        x = F.relu(self.BN2d3(self.conv3(x)))
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        return x

class MADL_classifier_dir(nn.Module):
    def __init__(self,num_classes=1,opt=None):
        super(MADL_classifier_dir, self).__init__()
        
        # Initial Conv layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # Stride of 2 to reduce dimensions
        
        # Conv layers
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Another stride of 2
        
        # Batch normalization
        self.BN2d1 = nn.GroupNorm(8,16)
        self.BN2d2 = nn.GroupNorm(8,32)
        self.BN2d3 = nn.GroupNorm(8,64)
        
        # Max pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce spatial dimensions
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc_cfd = nn.Linear(3,64)
        self.fc1 = nn.Linear(64, num_classes)
        self.opt = opt

    def forward(self, x ,cfd):
        x = x.unsqueeze(1)  # Add a channel dimension: [batch, 1, 2048, 512]

        x = F.relu(self.BN2d1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.BN2d2(self.conv2(x)))
        
        x = F.relu(self.BN2d3(self.conv3(x)))
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        if self.opt['cfd_classifer']:
            x = x + self.fc_cfd(cfd)
        x = self.fc1(x)
        return x

class MADL_cfd(nn.Module):
    def __init__(self):
        super(MADL_cfd, self).__init__()
        
        # Initial Conv layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # Stride of 2 to reduce dimensions
        
        # Conv layers
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Another stride of 2
        
        # Batch normalization
        self.BN2d1 = nn.BatchNorm2d(16)
        self.BN2d2 = nn.BatchNorm2d(32)
        self.BN2d3 = nn.BatchNorm2d(64)
        
        # Max pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce spatial dimensions
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc1 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension: [batch, 1, 2048, 512]

        x = F.relu(self.BN2d1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.BN2d2(self.conv2(x)))
        
        x = F.relu(self.BN2d3(self.conv3(x)))
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        return x
# device = torch.device("cpu")
# encoder = ResEncoder().to(device)
# decoder = ResClassifier().to(device)
# sample_input = torch.rand((2, 1, 64, 128, 128)).to(device)
# print(sample_input.shape)
# x,feat_channel,feats=encoder(sample_input)
# print(x.shape)
# print(feat_channel)

# out = decoder(x,feats)

# print(out.shape)