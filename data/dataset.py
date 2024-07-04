from torch.utils.data import Dataset
import torch.optim as optim
import os
import SimpleITK as sitk
import torch.nn.functional as F
import numpy as np
import pandas as pd
class CTCTA_dataset_truefalse(Dataset):

    def __init__(self,imagepath,cfdpath) -> None:
        super().__init__()
        data_path = imagepath
        cfd_path = cfdpath
        print(f"data_path:{data_path},cfd_path:{cfdpath}")
        #/mnt/data/CT_CTA/data_truefalse_vresion1 , /mnt/data/CT_CTA/cfd_res/inout_ratio.csv
        
        data_dir = os.listdir(data_path)
        self.ct_array=[]
        self.cta_array=[]
        self.label_true=[]
        self.label_false=[]

        self.isad=[]
        self.oa=[]
        self.v_minus=[]
        self.wbeta=[]

        self.cfdres = pd.read_csv(cfd_path)
        def get_is_ad_value(name):
           
            return self.cfdres[self.cfdres['name'] == name]['isad_v3'].values[0]
        def get_v_minus(name):
            return self.cfdres[self.cfdres['name'] == name]['v_minus'].values[0]
        def get_oa(name):
            return self.cfdres[self.cfdres['name'] == name][' Orthogonality Angle [ degree ]'].values[0]
        def get_wbeta(name):
            return self.cfdres[self.cfdres['name'] == name][' Velocity w.Beta'].values[0]
        for index in data_dir:
            # if index in self.adlist and index not in self.xzlist:
            index_path = os.path.join(data_path,index)
            ct_path = os.path.join(index_path,'ct.nii.gz')
            cta_path = os.path.join(index_path,'cta.nii.gz')
            label_true_path = os.path.join(index_path,'true.nii.gz')
            label_false_path = os.path.join(index_path,'false.nii.gz')

            ct_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
            cta_array = sitk.GetArrayFromImage(sitk.ReadImage(cta_path))
            label_true_array = sitk.GetArrayFromImage(sitk.ReadImage(label_true_path))
            label_false_array = sitk.GetArrayFromImage(sitk.ReadImage(label_false_path))

            self.ct_array.append(ct_array)
            self.cta_array.append(cta_array)
            self.label_true.append(label_true_array)
            self.label_false.append(label_false_array)

            
            self.isad.append(get_is_ad_value(int(index)))
            self.oa.append(get_oa(int(index)))
            self.v_minus.append(get_v_minus(int(index)))
            self.wbeta.append(get_wbeta(int(index)))

        self.ct_array = np.array(self.ct_array)
        self.cta_array = np.array(self.cta_array)
        self.label_true = np.array(self.label_true)
        self.label_false = np.array(self.label_false)

        self.isad = np.array(self.isad)
        self.oa = np.array(self.oa)
        self.v_minus = np.array(self.v_minus)
        self.wbeta = np.array(self.wbeta)
        self.cfd = np.column_stack((self.v_minus,self.oa,self.wbeta))
        self.cfd = (self.cfd - self.cfd.min(axis=0)) / (self.cfd.max(axis=0) - self.cfd.min(axis=0))
        # print(self.ct_array.shape)

        self.ct_array = np.expand_dims(self.ct_array,axis=1)
        self.cta_array = np.expand_dims(self.cta_array,axis=1)
        self.label_true = np.expand_dims(self.label_true,axis=1)
        self.label_false = np.expand_dims(self.label_false,axis=1)
        self.merge_label = self.label_true + self.label_false*2
        print(self.merge_label.min(),self.merge_label.max())
        # print(self.merge_label.shape)
        # print(f"ct_array.shape:  {self.ct_array.shape},ct_array.min:{self.ct_array.min()},ct_array.max:{self.ct_array.max()}")
        # print(f"cta_array.shape:  {self.cta_array.shape},cta_array.min:{self.cta_array.min()},cta_array.max:{self.cta_array.max()}")
        # print(f"label_true.shape:  {self.label_true.shape}")
        # print(f"label_false.shape:  {self.label_false.shape}")
        # print(f"is_ad:{self.isad}")
        # print(f"cfd.shape:  {self.cfd.shape}")
        
    def __len__(self):
        return self.ct_array.shape[0]
    
    def __getitem__(self,idx):
        return self.ct_array[idx],self.cta_array[idx],self.merge_label[idx],self.isad[idx],self.cfd[idx]


# dataset = CTCTA_dataset_truefalse(imagepath='/mnt/data/CT_CTA/data_truefalse_vresion1/train',cfdpath='/mnt/data/CT_CTA/cfd_res/inout_ratio.csv')
class CustomDataset_test(Dataset):
    def __init__(self,listad,listnad) -> None:
        super().__init__()
        self.ct_array=[]
        self.cta_array=[]
        self.label=[]
        for index in listad:
            ct_path = os.path.join(index,'ct.nii.gz')
            cta_path = os.path.join(index,'cta.nii.gz')
            ct_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
            cta_array = sitk.GetArrayFromImage(sitk.ReadImage(cta_path))
         
            self.ct_array.append(ct_array)
            self.cta_array.append(cta_array)
            self.label.append(1)
        
        for index in listnad:
            ct_path = os.path.join(index,'ct.nii.gz')
            cta_path = os.path.join(index,'cta.nii.gz')
            ct_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
            cta_array = sitk.GetArrayFromImage(sitk.ReadImage(cta_path))
         
            self.ct_array.append(ct_array)
            self.cta_array.append(cta_array)
            self.label.append(0)
        
        self.ct_array = np.array(self.ct_array)
        self.cta_array = np.array(self.cta_array)
        self.label = np.array(self.label)

        self.ct_array = np.expand_dims(self.ct_array,axis=1)
        self.cta_array = np.expand_dims(self.cta_array,axis=1)
       

        print(f"ct_array.shape:  {self.ct_array.shape},ct_array.min:{self.ct_array.min()},ct_array.max:{self.ct_array.max()}")
        print(f"cta_array.shape:  {self.cta_array.shape},cta_array.min:{self.cta_array.min()},cta_array.max:{self.cta_array.max()}")
        print(self.label)
    def __len__(self):
        return len(self.ct_array)
    
    def __getitem__(self,idx):
        return self.ct_array[idx],self.cta_array[idx],self.label[idx]



class CustomDataset_aorta(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        data_path = path
        print(f"data_path:{data_path}")
        data_dir = os.listdir(data_path)
        self.ct_array=[]
        self.cta_array=[]
        self.label=[]
        self.scope_array=[]
        self.adlist=np.load('/home/dingzhengyao/CT_CTA/2016_2017_crop_data/data_explain/16_19_ad.npy')
        self.nadlist=np.load('/home/dingzhengyao/CT_CTA/2016_2017_crop_data/data_explain/16_19_nad.npy')
        self.xzlist=np.load('/home/dingzhengyao/CT_CTA/2016_2017_crop_data/data_explain/16_19_xz.npy')
        for index in data_dir:
            index_path = os.path.join(data_path,index)
            ct_path = os.path.join(index_path,'ct.nii.gz')
            cta_path = os.path.join(index_path,'cta.nii.gz')
           
            ct_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
            cta_array = sitk.GetArrayFromImage(sitk.ReadImage(cta_path))
         

            self.ct_array.append(ct_array)
            self.cta_array.append(cta_array)
          
            self.label.append(1 if index in self.adlist and index not in self.xzlist else 0 )#ad:label 1

        self.ct_array = np.array(self.ct_array)
        self.cta_array = np.array(self.cta_array)
       
        self.label = np.array(self.label)

        self.ct_array = np.expand_dims(self.ct_array,axis=1)
        self.cta_array = np.expand_dims(self.cta_array,axis=1)
       

        print(f"ct_array.shape:  {self.ct_array.shape},ct_array.min:{self.ct_array.min()},ct_array.max:{self.ct_array.max()}")
        print(f"cta_array.shape:  {self.cta_array.shape},cta_array.min:{self.cta_array.min()},cta_array.max:{self.cta_array.max()}")
        
    def __len__(self):
        return len(self.ct_array)
    
    def __getitem__(self,idx):
        return self.ct_array[idx],self.cta_array[idx],self.label[idx]


class CTCTA_dataset_truefalse_new(Dataset):

    def __init__(self,data) -> None:
        super().__init__()
        
        self.ct_array = data["ct_array"]
        self.cta_array = data["cta_array"]
        self.merge_label = data["merge_label"]
        self.isad = data["isad"]
        self.cfd = data["cfd"]
        # print(f"ct_array.shape:  {self.ct_array.shape},ct_array.min:{self.ct_array.min()},ct_array.max:{self.ct_array.max()}")
        # print(f"cta_array.shape:  {self.cta_array.shape},cta_array.min:{self.cta_array.min()},cta_array.max:{self.cta_array.max()}")
        # print(f"merge_label.shape:  {self.merge_label.shape},merge_label.min:{self.merge_label.min()},merge_label.max:{self.merge_label.max()}")
        # print(f"is_ad:{self.isad}")
        # print(f"cfd.shape:  {self.cfd}")
    def __len__(self):
        return self.ct_array.shape[0]
    
    def __getitem__(self,idx):
        return self.ct_array[idx],self.cta_array[idx],self.merge_label[idx],self.isad[idx],self.cfd[idx]
