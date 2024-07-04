import torch.nn as nn
import torch.nn.functional as F
import torch

# adapt from https://github.com/MIC-DKFZ/BraTS2017

class CrossAttention(nn.Module):
    def __init__(self, physical_feature_dim, feature_dim, key_value_dim):
        super(CrossAttention, self).__init__()
        
        self.W_q = nn.Linear(physical_feature_dim, key_value_dim)
        self.W_k = nn.Conv3d(feature_dim, key_value_dim, kernel_size=1)
        self.W_v = nn.Conv3d(feature_dim, key_value_dim, kernel_size=1)
        self.adjustment_layer = nn.Linear(key_value_dim, feature_dim)
        self.merge = nn.Conv3d(feature_dim*2, feature_dim, kernel_size=1)
        self.feature_dim = feature_dim
    def forward(self, physical_features, ct_data):
        batch, _, depth, height, width = ct_data.size()
        
        # Generate Q, K, V
        Q = self.W_q(physical_features)  # batch x feature_dim
        K = self.W_k(ct_data).view(batch, -1, depth * height * width).permute(0, 2, 1)  # batch x feature_dim x (depth*height*width)
        V = self.W_v(ct_data).view(batch, -1, depth * height * width).permute(0, 2, 1)  # batch x feature_dim x (depth*height*width)
        # print(f"Q.shape:{Q.shape}")
        # print(f"K.shape:{K.shape}")
        # print(f"V.shape:{V.shape}")
        # print(f"Q.view(batch, 1, -1).shape:{Q.view(batch, 1, -1).shape}")
        # Attention
        attention_weights = F.softmax(Q.view(batch, 1, -1) @ K.transpose(-2, -1) / (K.size(-1) ** 0.5), dim=-1)
        # print(attention_weights.shape)
        # attn_weights = F.softmax(Q.view(batch, 1, -1) @ K.transpose(1, 2), dim=-1)  # batch x 1 x (depth*height*width)
        attn_output = attention_weights @ V
        # print(attn_output.shape)
        attn_output = self.adjustment_layer(attn_output.squeeze(1)).view(batch, self.feature_dim, 1, 1, 1)

        # Upsampling to match CT dimensions
        upsample = nn.Upsample(size=(depth, height, width), mode='trilinear')
        attn_output = upsample(attn_output)
        combined_features = torch.cat([ct_data, attn_output], dim=1)
        return self.merge(combined_features)


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

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
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



class Unet(nn.Module):
    def __init__(self, in_channels=4, base_channels=16, num_classes=4,opt=None):
        super(Unet, self).__init__()
        self.opt = opt
        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)
        self.attention1 = CrossAttention(physical_feature_dim=opt['pyhsical_dim'],feature_dim=base_channels*2,key_value_dim=opt['cross_attention_dim'])

        self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)
        self.attention2 = CrossAttention(physical_feature_dim=opt['pyhsical_dim'],feature_dim=base_channels*4,key_value_dim=opt['cross_attention_dim'])


        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)
        self.attention3 = CrossAttention(physical_feature_dim=opt['pyhsical_dim'],feature_dim=base_channels*8,key_value_dim=opt['cross_attention_dim'])


        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

    def forward(self, x,cfd):
        x = self.InitConv(x)       # (1, 16, 128, 128, 128)

        x1_1 = self.EnBlock1(x)
        x1_2 = self.EnDown1(x1_1)  # (1, 32, 64, 64, 64)
        if self.opt['cfd_embedding']:
            x1_2 = self.attention1(cfd,x1_2)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2 = self.EnDown2(x2_1)  # (1, 64, 32, 32, 32)
        if self.opt['cfd_embedding']:
            x2_2 = self.attention2(cfd,x2_2)

        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_2 = self.EnDown3(x3_1)  # (1, 128, 16, 16, 16)
        if self.opt['cfd_embedding']:
            x3_2 = self.attention3(cfd,x3_2)

        x4_1 = self.EnBlock4_1(x3_2)
        x4_2 = self.EnBlock4_2(x4_1)
        x4_3 = self.EnBlock4_3(x4_2)
        output = self.EnBlock4_4(x4_3)  # (1, 128, 16, 16, 16)

        return x1_1,x2_1,x3_1,output


if __name__ == '__main__':
    with torch.no_grad():
        import os
        version = 'version40'
    #     opt = {
    #     "lamda_G_L1":10,
    #     "lamda_G_per":10,
    #     "lamda_G_seg":100,
    #     "lamda_G_CE":10,
    #     "epoch":199,
    #     "describe":version,
    #     "device":"cuda:2",
    #     "batch_size":2,
    #     "lr_g":0.00001,
    #     # "lr_g":{
    #     #     "encoder":0.00005,
    #     #     "decoder1":0.0001,
    #     #     "decoder2":0.0002
    #     # },
    #     "lr_c":0.00001,
    #     "lr_d":0.0001,
    #     "lr_cfd":0.00001,
    #     "norm":'unet:gn,trans:gn,classifier:gn',
    #     "classification_threshold":0.5,
    #     "cfd_embedding":True,
    #     "include_background":False,
    #     "seed_value" : 42,
    #     "pretrained_cfd":True,
    #     "cfd_classifer":True,
    #     "train_ratio":5 ,
    #     "traincfd_epoch":199,
    #     "version":version,
    #     "pyhsical_dim":2,
    #     "cross_attention_dim":32
    # }
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 1, 64, 128, 128), device=cuda0)
        cfd = torch.rand((1, 2), device=cuda0)
        # model = Unet1(in_channels=4, base_channels=16, num_classes=4)
        model = Unet(in_channels=1, base_channels=16, num_classes=4,opt=opt)
        model.cuda()
        output = model(x,cfd)
        print('output:', output[3].shape)