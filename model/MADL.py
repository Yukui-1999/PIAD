import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn.functional as F
from model.net import Discriminator,GANLoss,MADL_cfd,MADL_classifier,MADL_classifier_dir,Small3DCNN
from model.lpips import LPIPS
from model.TransUnet_cfd.TransBTS_downsample8x_skipconnection import TransBTS_cfd
from torch.optim.lr_scheduler import LambdaLR
from model.TransUnet.TransBTS_downsample8x_skipconnection import TransBTS
import monai
import os
from monai.losses import DiceCELoss, GeneralizedDiceLoss, DiceFocalLoss
class MADL(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,device=None,opt=None,cfdmodel=None):
        super(MADL, self).__init__()
        self.device = device
        self.opt = opt
        self.lamda_G_L1 = opt['lamda_G_L1']
        self.lamda_G_per = opt['lamda_G_per']
        self.lamda_G_seg = opt['lamda_G_seg']
        self.lamda_G_CE = opt['lamda_G_CE']
        self.batchsize = opt['batch_size']
        self.midepoch = (opt['epoch'] + 1)/2
        self.pretrained_cfd = opt['pretrained_cfd']
        self.cfd_classifer = opt["cfd_classifer"]

        # self.classifier_post = Small3DCNN(num_classes=1)
        # self.classifier_post = self.classifier_post.to(self.device)

        _,self.generator = TransBTS(opt=opt)
        self.generator = self.generator.to(self.device)

        self.discriminator = Discriminator(in_channels=out_channels,n=64)
        self.discriminator = self.discriminator.to(self.device)

        
        self.classifier = MADL_classifier_dir(num_classes=1,opt=opt)
        self.classifier = self.classifier.to(self.device)

        
        if self.pretrained_cfd:
            self.cfdpredict = cfdmodel.eval()
        else:
            _ , self.cfdpredict = TransBTS_cfd()
            self.cfdpredict = self.cfdpredict.to(self.device)
        

        self.perceptual_model = LPIPS().to(self.device).eval()

        self.criterionGAN = GANLoss().to(self.device)
        self.criterionC = torch.nn.BCEWithLogitsLoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionSeg = GeneralizedDiceLoss(
            include_background=opt['include_background'],  # 考虑背景类别
            to_onehot_y=True,  # 转换标签为 one-hot 格式
            softmax=True,  # 使用 Softmax 激活
        )
        
        self.optimizer_G = optim.AdamW(self.generator.parameters(), lr=opt['lr_g'], betas=(0.5, 0.999))
        self.optimizer_D = optim.AdamW(self.discriminator.parameters(), lr=opt['lr_d'], betas=(0.5, 0.999))
        self.optimizer_C = optim.AdamW(self.classifier.parameters(), lr=opt['lr_c'], betas=(0.5, 0.999))
        self.optimizer_cfd = optim.AdamW(self.cfdpredict.parameters(), lr=opt['lr_cfd'], betas=(0.5, 0.999))
        self.optimizers = {
                    'G': self.optimizer_G,
                    'D': self.optimizer_D,
                    'C': self.optimizer_C,
                    'cfd': self.optimizer_cfd
                }
        self.schedulers = {}
        for name, optimizer in self.optimizers.items():
            self.schedulers[name] = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - (epoch - self.midepoch) / self.midepoch if epoch >= self.midepoch else 1)
        
    def set_input(self, ct , real_cta ,label ,isad ,cfd):
        self.ct = ct.float().to(self.device)
        self.real_cta = real_cta.float().to(self.device)
        if not self.opt['Test']:
            self.label = label.float().to(self.device)
        self.isad = isad.float().to(self.device)
        self.isad = self.isad.unsqueeze(1)
        # self.cfd = torch.rand((self.batchsize, 2), device=self.device)
        
        self.cfd_label = cfd.float().to(self.device)


    def forward(self):
        # print(f'self.ct.shape:{self.ct.shape}')
        # print(f'self.cfd.shape:{self.cfd.shape}')
        if self.opt['cfd_embedding']:
            if self.pretrained_cfd:
                with torch.no_grad():
                    self.cfd = self.cfdpredict(self.ct)
            else:
                self.cfd = self.cfdpredict(self.ct)
            # print(f'self.cfd:{self.cfd.shape}')
            self.ct_deepcopy = self.ct.clone()
            # print('cfd enable')
            assert self.opt['cfd_embedding'] 
            self.generate_segment ,self.generate_cta ,self.encoder_output = self.generator(self.ct_deepcopy ,self.cfd.detach())
        else:
            if self.pretrained_cfd and self.cfd_classifer: 
                with torch.no_grad():
                    self.cfd = self.cfdpredict(self.ct)
            elif not self.pretrained_cfd and self.cfd_classifer:
                self.cfd = self.cfdpredict(self.ct)
            else:
                self.cfd = torch.zeros_like(self.cfd_label)
                self.cfd_label = torch.zeros_like(self.cfd)
            # print('cfd not enable')
            assert not self.opt['cfd_embedding']
            self.generate_segment ,self.generate_cta ,self.encoder_output = self.generator(self.ct ,torch.zeros_like(self.cfd_label))
        # print(f'self.generate_segment:{self.generate_segment.shape}')
        # print(f'self.generate_cta:{self.generate_cta.shape}')
        # print(f'self.encoder_output:{self.encoder_output.shape}')
              
    def backward_C(self,compute_gradients=True):
        if self.opt['classifier_post']:
            self.pred_ad = self.classifier_post(torch.cat((self.generate_segment.detach(), self.generate_cta.detach()), dim=1))
            # print(f'self.pred_ad_output:{self.pred_ad.shape}')
            self.loss_C = self.criterionC(self.pred_ad, self.isad)
            if compute_gradients:
                self.loss_C.backward()
        else:  
            self.pred_ad = self.classifier(self.encoder_output.detach(),self.cfd.detach())
            # print(f'self.pred_ad_output:{self.pred_ad.shape}')
            self.loss_C = self.criterionC(self.pred_ad, self.isad)
            if compute_gradients:
                self.loss_C.backward()
    
    def backward_cfd(self,compute_gradients=True):
       
        self.loss_cfd = self.criterionL1(self.cfd,self.cfd_label)
        if compute_gradients:
            self.loss_cfd.backward()
        

    def backward_D(self, compute_gradients=True):
       

        pred_fake = self.discriminator(self.ct, self.generate_cta.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
      
        pred_real = self.discriminator(self.ct, self.real_cta)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        if compute_gradients:
            self.loss_D.backward()

    def backward_G(self, compute_gradients=True):
        
        # generate cta
        pred_fake = self.discriminator(self.ct, self.generate_cta)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
     
        self.loss_G_L1 = self.criterionL1(self.generate_cta, self.real_cta)
        
        current_batch_size1 = self.generate_cta.shape[0]
        current_batch_size2 = self.real_cta.shape[0]
        generate_cta_reshaped = self.generate_cta.view(current_batch_size1*64, 1, 128, 128)
        real_cta_reshaped = self.real_cta.view(current_batch_size2*64, 1, 128, 128)
        self.perceptual_loss = self.perceptual_model(generate_cta_reshaped, real_cta_reshaped).mean()

        # segment
        output_permuted = self.generate_segment.permute(0, 1, 3, 4, 2)
        labels_permuted = self.label.permute(0, 1, 3, 4, 2)
        # monai GeneralizedDiceLoss Args:
        # input: the shape should be BNH[WD].
        # target: the shape should be BNH[WD].
        self.loss_G_seg = self.criterionSeg(output_permuted,labels_permuted)
        if self.opt['classifier_post']:
            self.pred_ad = self.classifier_post(torch.cat((self.generate_segment, self.generate_cta), dim=1))
            # print(f'self.pred_ad_output:{self.pred_ad.shape}')
            self.loss_G_CE = self.criterionC(self.pred_ad, self.isad)
        else:  
            # classifier
            pred_label = self.classifier(self.encoder_output,self.cfd.detach())
            self.loss_G_CE = self.criterionC(pred_label,self.isad)


        self.loss_G = (self.loss_G_GAN + 
               self.loss_G_L1 * self.lamda_G_L1 + 
               self.perceptual_loss * self.lamda_G_per +
               self.loss_G_seg * self.lamda_G_seg + 
               self.loss_G_CE * self.lamda_G_CE 
               )
        if compute_gradients:
            self.loss_G.backward()
        
    def set_requires_grad(self, nets, requires_grad=False):
       
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def log(self,batchidx,logfile,wandb):
        print(f"batch_idx:{batchidx}正常运行,loss_D:{self.loss_D},loss_G:{self.loss_G},\
              loss_C:{self.loss_C},loss_cfd:{ self.loss_cfd}")
        logfile.write(f"batch_idx:{batchidx}正常运行,loss_D:{self.loss_D},\
                      loss_G:{self.loss_G},loss_C:{self.loss_C},loss_cfd:{ self.loss_cfd}\n")
        logfile.flush()
        current_log = {
            "loss_D":self.loss_D,
            "loss_G":self.loss_G,
            "loss_C":self.loss_C,
            "loss_cfd": self.loss_cfd,
            "loss_d_fake":self.loss_D_fake,
            "loss_d_real":self.loss_D_real,
            "loss_g_gan":self.loss_G_GAN,
            "loss_g_L1":self.loss_G_L1,
            "loss_g_per":self.perceptual_loss,
            "loss_g_ce":self.loss_G_CE,
            "loss_g_seg":self.loss_G_seg
        }
        wandb.log(current_log)

        print(f"loss_d_fake:{self.loss_D_fake},loss_d_real:{self.loss_D_real},\
              loss_g_gan:{self.loss_G_GAN},loss_g_L1:{self.loss_G_L1},loss_g_per:{self.perceptual_loss},\
                loss_g_ce:{self.loss_G_CE},loss_g_seg:{self.loss_G_seg}")
        logfile.write(f"loss_d_fake:{self.loss_D_fake},loss_d_real:{self.loss_D_real},\
              loss_g_gan:{self.loss_G_GAN},loss_g_L1:{self.loss_G_L1},loss_g_per:{self.perceptual_loss},\
                loss_g_ce:{self.loss_G_CE},loss_g_seg:{self.loss_G_seg}")
        logfile.flush()
        return current_log
        
    def save_model(self,epoch,path):
        state_dicts = {'epoch': epoch, 'model_state_dict': self.state_dict()}

        for name, optimizer in self.optimizers.items():
            state_dicts[name + '_state_dict'] = optimizer.state_dict()

        for name, scheduler in self.schedulers.items():
            state_dicts[name + '_state_dict'] = scheduler.state_dict()

        torch.save(state_dicts, path)
    
    def load_model_from_checkpoint(self, checkpoint_path):
        # print(checkpoint_path)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            # Load model state
            self.load_state_dict(checkpoint['model_state_dict'])

            # Load states for optimizers
            for name, optimizer in self.optimizers.items():
                if name + '_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint[name + '_state_dict'])

            # Load states for schedulers
            for name, scheduler in self.schedulers.items():
                if name + '_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint[name + '_state_dict'])

            # Load epoch
            start_epoch = checkpoint['epoch']
            print("Loaded checkpoint from epoch:", start_epoch)
        else:
            print("No checkpoint found at:", checkpoint_path)
            start_epoch = 0  # 如果没有找到检查点，从零开始

        return start_epoch

    def optimize_parameters(self,batch_idx):
        self.forward()
        Cumulative_gradient = 1 #累计梯度用
        train_ratio = self.opt["train_ratio"]      
        if batch_idx % Cumulative_gradient == 0:# compute fake images: G(A)     
            if batch_idx % train_ratio == 0:
                # update D
                self.set_requires_grad(self.discriminator, True)  # enable backprop for D
                self.backward_D()                # calculate gradients for D
                self.optimizer_D.step()          # update D's
                self.optimizer_D.zero_grad()     # set D's gradients to zero
            if not self.pretrained_cfd and self.opt['cfd_embedding']:
                # update cfd_predict
                self.backward_cfd()
                self.optimizer_cfd.step()
                self.optimizer_cfd.zero_grad()
            else :
                self.backward_cfd(compute_gradients=False)
            # update C
            self.set_requires_grad(self.classifier, True)
            self.backward_C()
            self.optimizer_C.step()
            self.optimizer_C.zero_grad()
            # update G
            self.set_requires_grad(self.discriminator, False) 
            self.set_requires_grad(self.classifier, False)

            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # update G's weights
            self.optimizer_G.zero_grad()        # set G's gradients to zero
           
        else:
            print('Cumulative_gradient...')
            if batch_idx%5==0:
                # update D
                self.set_requires_grad(self.discriminator, True)  # enable backprop for D
                self.backward_D()                # calculate gradients for D
            # update cfd
            if not self.pretrained_cfd:
                self.backward_cfd()
            else :
                self.backward_cfd(compute_gradients=False)
            # update C
            self.set_requires_grad(self.classifier, True)
            self.backward_C()
            # update G
            self.set_requires_grad(self.discriminator, False)  # D requires no gradients when optimizing G
            self.set_requires_grad(self.classifier, False)
            self.backward_G()                   # calculate graidents for G
          