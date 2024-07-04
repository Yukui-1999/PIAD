import torch
import torch.nn as nn
from torchinfo  import summary
from model.TransUnet.Transformer import TransformerModel
from model.TransUnet.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from model.TransUnet.unet import Unet

class encoder(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
        opt=None
    ):
        super(encoder, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:

            self.conv_x = nn.Conv3d(
                128,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.Unet = Unet(in_channels=1, base_channels=16, num_classes=4,opt=opt)
        self.bn = nn.GroupNorm(8,128)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)

    def forward(self, x, cfd):
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution
            x1_1, x2_1, x3_1, x = self.Unet(x,cfd)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)

        else:
            x = self.Unet(x)
            x = self.bn(x)
            x = self.relu(x)
            x = (
                x.unfold(2, 2, 2)
                .unfold(3, 2, 2)
                .unfold(4, 2, 2)
                .contiguous()
            )
            x = x.view(x.size(0), x.size(1), -1, 8)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        return x1_1, x2_1, x3_1, x, intmd_x

class decoder(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        task,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",   
    ):
        super(decoder, self).__init__()
        self.task = task
        self.outfeature = 3 if task == "seg" else 1
        self.embedding_dim = embedding_dim
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.Softmax = nn.Softmax(dim=1)
        self.Tanh = nn.Tanh()

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim//4, out_channels=self.embedding_dim//8)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim//8)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim//8, out_channels=self.embedding_dim//16)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim//16)

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim//16, out_channels=self.embedding_dim//32)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim//32)

        self.endconv = nn.Conv3d(self.embedding_dim // 32, self.outfeature, kernel_size=1)
    
    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim/2 / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

    def forward(self,x1_1, x2_1, x3_1, x, intmd_x, intmd_layers=[1, 2, 3, 4]):
        assert intmd_layers is not None, "pass the intermediate layers for MLA"
        encoder_outputs = {}
        all_keys = []
        for i in intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()
      
        x8 = encoder_outputs[all_keys[0]]
        x8 = self._reshape_output(x8)
        x8 = self.Enblock8_1(x8)
        x8 = self.Enblock8_2(x8)

        y4 = self.DeUp4(x8, x3_1)  # (1, 64, 32, 32, 32)
        y4 = self.DeBlock4(y4)

        y3 = self.DeUp3(y4, x2_1)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)

        y2 = self.DeUp2(y3, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)

        y = self.endconv(y2)      # (1, 4, 128, 128, 128)

        if self.task != "seg":
            y = self.Tanh(y)
        # y = self.Softmax(y)
        return y



class TransformerBTS(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
        opt=None
    ):
        super(TransformerBTS, self).__init__()
        self.encode = encoder(img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",
            opt=opt)
        self.decode1 = decoder(img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            task="seg",
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",  
            )
        
        self.decode2 = decoder(img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            task="gen",
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",  
            )
        self.encoder_lr = opt["lr_g"]["encoder"]
        self.decoder1_lr = opt["lr_g"]["decoder1"]
        self.decoder2_lr = opt["lr_g"]["decoder2"]
        

    def build_optimizers(self):
        # Encoder parameters
        encoder_params = self.encode.parameters()

        # Decoder1 parameters
        decoder1_params = self.decode1.parameters()

        # Decoder2 parameters
        decoder2_params = self.decode2.parameters()

        # Create optimizers
        encoder_optimizer = torch.optim.AdamW(encoder_params, lr=self.encoder_lr,betas=(0.5, 0.999), weight_decay=0.001)
        decoder1_optimizer = torch.optim.AdamW(decoder1_params, lr=self.decoder1_lr,betas=(0.5, 0.999), weight_decay=0.001)
        decoder2_optimizer = torch.optim.AdamW(decoder2_params, lr=self.decoder2_lr,betas=(0.5, 0.999), weight_decay=0.001)

        return encoder_optimizer, decoder1_optimizer, decoder2_optimizer

    def forward(self, x, cfd,auxillary_output_layers=[1, 2, 3, 4]):

        x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs = self.encode(x,cfd)
        # print(f'encoder_output.shape:{encoder_output.shape}')
        # print(f'intmd_encoder_outputs:{intmd_encoder_outputs["0"].shape}')
        decoder_output = self.decode1(
            x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )
        decoder_output_d2 = self.decode2(
            x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]

            return decoder_output,decoder_output_d2,encoder_output

        return decoder_output,decoder_output_d2,encoder_output

    

class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.GroupNorm(8,512 // 4)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.bn2 = nn.GroupNorm(8,512 // 4)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8,512 // 4)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.bn2 = nn.GroupNorm(8,512 // 4)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.GroupNorm(8,in_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8,in_channels)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01,inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1




def TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned",opt=None):

    if dataset.lower() == 'brats':
        img_dim = 128
        num_classes = 4

    num_channels = 1
    patch_dim = 8
    aux_layers = [1, 2, 3, 4]
    model = TransformerBTS(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=512,
        num_heads=8,
        num_layers=4,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
        opt=opt
    )

    return aux_layers, model


if __name__ == '__main__':
    with torch.no_grad():
        import os
        version = 'version40'
        opt = {
        "lamda_G_L1":10,
        "lamda_G_per":10,
        "lamda_G_seg":100,
        "lamda_G_CE":10,
        "epoch":199,
        "describe":version,
        "device":"cuda:2",
        "batch_size":2,
        "lr_g":0.00001,
        # "lr_g":{
        #     "encoder":0.00005,
        #     "decoder1":0.0001,
        #     "decoder2":0.0002
        # },
        "lr_c":0.00001,
        "lr_d":0.0001,
        "lr_cfd":0.00001,
        "norm":'unet:gn,trans:gn,classifier:gn',
        "classification_threshold":0.5,
        "cfd_embedding":True,
        "include_background":False,
        "seed_value" : 42,
        "pretrained_cfd":True,
        "cfd_classifer":True,
        "train_ratio":5 ,
        "traincfd_epoch":199,
        "version":version,
        "pyhsical_dim":2,
        "cross_attention_dim":32
        }
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((2, 1, 64, 128, 128), device=cuda0)
        cfd = torch.rand((2, 3), device=cuda0)
        _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned",opt=opt)
        model.cuda()
        y1,y2,y3 = model(x,cfd)
        print(y1.shape)
        print(y2.shape)
        print(y3.shape)
        summary(model,input_size=((1, 1, 64, 128, 128),(1,3)), verbose=2)
        # export PYTHONPATH=$PYTHONPATH:~/CT_CTA/Final_ctcta
