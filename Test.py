from torch.utils.data import  DataLoader,random_split
import torch
from model.MADL import MADL
from data.dataset import CTCTA_dataset_truefalse,CTCTA_dataset_truefalse_new,CustomDataset_test
from sklearn.metrics import classification_report
import numpy as np
import os
import wandb
import SimpleITK as sitk
import random
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from util import compute_binary_classification_metrics,import_data,get_train_test_sets,train_cfd,cfd_model
from model.TransUnet_cfd.TransBTS_downsample8x_skipconnection import TransBTS_cfd
import re
import pandas as pd
from util import calculate_ssim_psnr,multi_class_dice_score_jaccard_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

seed = 473
random.seed(seed)
os.environ['PYTHONHASHSEED'] =str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic =True

adlist = np.load("/home/dingzhengyao/CT_CTA/2016_2017_crop_data/data_explain/16_20_ad.npy")
nadlist = np.load("/home/dingzhengyao/CT_CTA/2016_2017_crop_data/data_explain/16_20_nad.npy")

sample_adlist = list(adlist)
sample_nadlist = list(nadlist)

listversion = [["version49_","version48_"],
               ["version51_","version50_"],
               ["version59_","version58_"],
               ["version61_","version60_"],
               ["version57_","version64_"]]

dataset = CustomDataset_test(sample_adlist,sample_nadlist)
version = 'none'



def net_benefit(tp, fp, tn, fn, threshold):
    benefit = tp - fp * threshold / (1 - threshold)
    return benefit / (tp + fp + tn + fn)

def decision_curve_analysis(y_true, y_prob, thresholds=np.arange(0.01, 1.00, 0.01)):
    nb = []
    y_prob = y_prob.flatten()
    print(f"y_prob.len:{len(y_prob)}")
    print(f"y_true.len:{len(y_true)}")
    

    print(y_prob)
    print(y_true)
    for th in thresholds:
        tp = np.sum((y_prob >= th) & (y_true == 1))
        print(f"tp:{tp}")
        fp = np.sum((y_prob >= th) & (y_true == 0))
        print(f"fp:{fp}")
        tn = np.sum((y_prob < th) & (y_true == 0))
        print(f"tn:{tn}")
        fn = np.sum((y_prob < th) & (y_true == 1))
        print(f"fn:{fn}")
        nb.append(net_benefit(tp, fp, tn, fn, th))
        # print(f"th:{th},tp:{tp},fp:{fp},tn:{tn},fn:{fn},")
    return thresholds, nb


for j in range(5):
    acc1=[]
    auc1=[]
    f11=[]
    precision1=[]
    recall1=[]
    spec1=[]
    ssim1=[]
    psrn1=[]
    for i in range(5):
        opt1 = {
        "lamda_G_L1":10,
        "lamda_G_per":10,
        "lamda_G_seg":100,
        "lamda_G_CE":10,
        "epoch":199,
        "describe":version,
        "device":"cuda:3",
        "batch_size":2,
        "lr_g":0.00001,
        "lr_c":0.00001,
        "lr_d":0.0001,
        "lr_cfd":0.00001,
        "norm":'unet:gn,trans:gn,classifier:gn',
        "classification_threshold":0.4,
        "cfd_embedding":True,
        "include_background":False,
        # "seed_value" : seed,
        "pretrained_cfd":True,
        "cfd_classifer":False,
        "train_ratio":5 ,
        "traincfd_epoch":80,
        "version":listversion[j][0],
        "pyhsical_dim":3,
        "cross_attention_dim":32,
        "Test":True
    }
        device = torch.device( opt1['device'] if torch.cuda.is_available() else 'cpu')
        test_dataloader = DataLoader(dataset, batch_size=opt1['batch_size'], shuffle=True)
        checkpoint = torch.load("/mnt/data/CT_CTA/Final_ctcta/trained_model/"+opt1['version']+str(i)+"/weight_cfd.pth")
        _ , cfdmodel = TransBTS_cfd()
        cfdmodel.load_state_dict(checkpoint)
        cfdmodel = cfdmodel.to(opt1["device"])

        best_model_path = "/mnt/data/CT_CTA/Final_ctcta/best_model/"+opt1['version']+str(i)+"/weight.pth"
        best_model = torch.load(best_model_path)
        model = MADL(device=device,opt=opt1,cfdmodel=cfdmodel).to(device)
        model.load_state_dict(best_model)
        print("loaded model")
        model.eval()
        with torch.no_grad():
            ssim = []
            psrn = []
            dice_score = []
            jaccard_score = []
            predicted_probs = torch.Tensor([]).float().to(device)
            pre_label = torch.Tensor([]).long().to(device)  # assuming you're using a device object for your model's device
            real_label = torch.Tensor([]).long()
            for batch_idx, (real_images, target_images,isad) in enumerate(test_dataloader):
                model.set_input(real_images,target_images,None,isad,torch.rand(opt1['batch_size'], 3))
                model.forward()  # compute predictions
                model.backward_C(compute_gradients=False)
                output_seg = model.generate_segment
                output_gen = model.generate_cta
                predicted_probs = torch.cat((predicted_probs,torch.sigmoid(model.pred_ad)),dim=0)
                pre_label = torch.cat((pre_label, (torch.sigmoid(model.pred_ad) > opt1['classification_threshold']).float()), dim=0)
                real_label = torch.cat((real_label, isad), dim=0)
                ssim_,psrn_ = calculate_ssim_psnr(output_gen.cpu(),target_images)
                
                ssim.append(ssim_)
                psrn.append(psrn_)
            # 计算验证集上的平均损失
            thresholds, nb = decision_curve_analysis(real_label.cpu().numpy(), predicted_probs.cpu().numpy())
            dca_data = pd.DataFrame({'Threshold': thresholds, 'Net Benefit': nb})
            dca_data.to_csv(f"/home/dingzhengyao/CT_CTA/Final_ctcta/Test_result04/DCA_data_{opt2['version']}_{i}.csv", index=False)

            # 在for循环内部，计算每一折的ROC和PRC
            fpr, tpr, _ = roc_curve(real_label.cpu().numpy(), predicted_probs.cpu().numpy())
            roc_auc = auc(fpr, tpr)

            precision, recall, _ = precision_recall_curve(real_label.cpu().numpy(), predicted_probs.cpu().numpy())
            prc_auc = average_precision_score(real_label.cpu().numpy(), predicted_probs.cpu().numpy())
            prob_true, prob_pred = calibration_curve(real_label.cpu().numpy(), predicted_probs.cpu().numpy(), n_bins=20)

           
            
            # 将ROC和PRC数据及其AUC保存到DataFrame中
            roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'ROC AUC': [roc_auc] * len(fpr)})
            prc_data = pd.DataFrame({'Recall': recall, 'Precision': precision, 'PRC AUC': [prc_auc] * len(recall)})
            calibration_data = pd.DataFrame({'Prob_True': prob_true, 'Prob_Pred': prob_pred})
            # 保存到CSV文件
            calibration_data.to_csv(f"/home/dingzhengyao/CT_CTA/Final_ctcta/Test_result04/Calibration_data_{opt2['version']}_{i}.csv", index=False)
            roc_data.to_csv(f"/home/dingzhengyao/CT_CTA/Final_ctcta/Test_result04/ROC_data_{opt2['version']}_{i}.csv", index=False)
            prc_data.to_csv(f"/home/dingzhengyao/CT_CTA/Final_ctcta/Test_result04/PRC_data_{opt2['version']}_{i}.csv", index=False)

 
            # 计算验证集上的平均损失

            metrics = compute_binary_classification_metrics(real_label.cpu().numpy(), pre_label.cpu().numpy(), predicted_probs.cpu().numpy())
            
            
            # 打印
            print_str = (f"Accuracy: {metrics['Accuracy']}, Precision: {metrics['Precision']}, "
                        f"Recall: {metrics['Recall']}, F1 Score: {metrics['F1 Score']}, AUC: {metrics['AUC']}")
            
            acc1.append(metrics['Accuracy'])
            precision1.append(metrics['Precision'])
            recall1.append(metrics['Recall'])
            f11.append(metrics['F1 Score'])
            auc1.append(metrics['AUC'])
            spec1.append(metrics['Specificity'])
            ssim1.append(np.mean(ssim))
            psrn1.append(np.mean(psrn))
            
            # 使用正则表达式提取数字
            print(print_str)

    avg_acc1 = sum(acc1) / len(acc1)
    avg_precision1 = sum(precision1) / len(precision1)
    avg_recall1 = sum(recall1) / len(recall1)
    avg_f11 = sum(f11) / len(f11)
    avg_auc1 = sum(auc1) / len(auc1)
    avg_spec1 = sum(spec1) / len(spec1)
    avg_ssim1 = sum(ssim1) / len(ssim1)
    avg_psrn1 = sum(psrn1) / len(psrn1)


    var_acc1 = np.var(acc1,ddof=1)
    var_precision1 = np.var(precision1,ddof=1)
    var_recall1 = np.var(recall1,ddof=1)
    var_f11 = np.var(f11,ddof=1)
    var_auc1 = np.var(auc1,ddof=1)
    var_spec1 = np.var(spec1, ddof=1)
    var_ssim1 = np.var(ssim1, ddof=1)
    var_psrn1 = np.var(psrn1, ddof=1)


    # 4. Save the metrics to a CSV file
    results_df1 = pd.DataFrame({
        'Fold': list(range(1, 6)) + ['Average', 'Variance'],
        'Accuracy': acc1 + [avg_acc1, var_acc1],
        'Precision': precision1 + [avg_precision1, var_precision1],
        'Recall': recall1 + [avg_recall1, var_recall1],
        'F1 Score': f11 + [avg_f11, var_f11],
        'AUC': auc1 + [avg_auc1, var_auc1],
        'Specificity': spec1 + [avg_spec1, var_spec1],
        'SSIM': ssim1 + [avg_ssim1, var_ssim1],
        'PSRN': psrn1 + [avg_psrn1, var_psrn1],
    })

    results_df1.to_csv("Test_result04/"+opt1['version']+"04.csv", index=False)
            


    acc2=[]
    auc2=[]
    f12=[]
    precision2=[]
    recall2=[]
    spec2=[]
    ssim2=[]
    psrn2=[]

    for i in range(5):
            
        opt2 = {
            "lamda_G_L1":10,
            "lamda_G_per":10,
            "lamda_G_seg":100,
            "lamda_G_CE":10,
            "epoch":199,
            "describe":version,
            "device":"cuda:3",
            "batch_size":2,
            "lr_g":0.00001,
            "lr_c":0.00001,
            "lr_d":0.0001,
            "lr_cfd":0.00001,
            "norm":'unet:gn,trans:gn,classifier:gn',
            "classification_threshold":0.45,
            "cfd_embedding":False,
            "include_background":False,
            # "seed_value" : seed,
            "pretrained_cfd":False,
            "cfd_classifer":False,
            "train_ratio":5 ,
            "traincfd_epoch":80,
            "version":listversion[j][1],
            "pyhsical_dim":3,
            "cross_attention_dim":32,
            "Test":True,
            "Ablation_transformer_use":True,
            "classifier_post":False,
        }
        device = torch.device( opt2['device'] if torch.cuda.is_available() else 'cpu')
        test_dataloader = DataLoader(dataset, batch_size=opt2['batch_size'], shuffle=True)
        # checkpoint = torch.load("/mnt/data/CT_CTA/Final_ctcta/trained_model/"+opt2['version']+str(i)+"/weight_cfd.pth")
        checkpoint = torch.load("/mnt/data/CT_CTA/Final_ctcta/trained_model/version47_"+str(i)+"/weight_cfd.pth")
        _ , cfdmodel = TransBTS_cfd()
        cfdmodel.load_state_dict(checkpoint)
        cfdmodel = cfdmodel.to(opt2["device"])

        best_model_path = "/mnt/data/CT_CTA/Final_ctcta/best_model/"+opt2['version']+str(i)+"/weight.pth"
        best_model = torch.load(best_model_path)
        model = MADL(device=device,opt=opt2,cfdmodel=cfdmodel).to(device)
        model.load_state_dict(best_model)
        print("loaded model")
        model.eval()
        with torch.no_grad():
            ssim = []
            psrn = []
            dice_score = []
            jaccard_score = []
            predicted_probs = torch.Tensor([]).float().to(device)
            pre_label = torch.Tensor([]).long().to(device)  # assuming you're using a device object for your model's device
            real_label = torch.Tensor([]).long()
            for batch_idx, (real_images, target_images,isad) in enumerate(test_dataloader):
                model.set_input(real_images,target_images,None,isad,torch.rand(opt2['batch_size'], 3))
                model.forward()  # compute predictions
                model.backward_C(compute_gradients=False)
                output_seg = model.generate_segment
                output_gen = model.generate_cta
                predicted_probs = torch.cat((predicted_probs,torch.sigmoid(model.pred_ad)),dim=0)
                pre_label = torch.cat((pre_label, (torch.sigmoid(model.pred_ad) > opt2['classification_threshold']).float()), dim=0)
                real_label = torch.cat((real_label, isad), dim=0)
                ssim_,psrn_ = calculate_ssim_psnr(output_gen.cpu(),target_images)
                
                ssim.append(ssim_)
                psrn.append(psrn_)
                
            # 计算验证集上的平均损失
            thresholds, nb = decision_curve_analysis(real_label.cpu().numpy(), predicted_probs.cpu().numpy())
            dca_data = pd.DataFrame({'Threshold': thresholds, 'Net Benefit': nb})
            dca_data.to_csv(f"/home/dingzhengyao/CT_CTA/Final_ctcta/Test_result04/DCA_data_{opt2['version']}_{i}.csv", index=False)

            # 在for循环内部，计算每一折的ROC和PRC
            fpr, tpr, _ = roc_curve(real_label.cpu().numpy(), predicted_probs.cpu().numpy())
            roc_auc = auc(fpr, tpr)

            precision, recall, _ = precision_recall_curve(real_label.cpu().numpy(), predicted_probs.cpu().numpy())
            prc_auc = average_precision_score(real_label.cpu().numpy(), predicted_probs.cpu().numpy())
            prob_true, prob_pred = calibration_curve(real_label.cpu().numpy(), predicted_probs.cpu().numpy(), n_bins=20)

           
            
            # 将ROC和PRC数据及其AUC保存到DataFrame中
            roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'ROC AUC': [roc_auc] * len(fpr)})
            prc_data = pd.DataFrame({'Recall': recall, 'Precision': precision, 'PRC AUC': [prc_auc] * len(recall)})
            calibration_data = pd.DataFrame({'Prob_True': prob_true, 'Prob_Pred': prob_pred})
            # 保存到CSV文件
            calibration_data.to_csv(f"/home/dingzhengyao/CT_CTA/Final_ctcta/Test_result04/Calibration_data_{opt2['version']}_{i}.csv", index=False)
            roc_data.to_csv(f"/home/dingzhengyao/CT_CTA/Final_ctcta/Test_result04/ROC_data_{opt2['version']}_{i}.csv", index=False)
            prc_data.to_csv(f"/home/dingzhengyao/CT_CTA/Final_ctcta/Test_result04/PRC_data_{opt2['version']}_{i}.csv", index=False)


            metrics = compute_binary_classification_metrics(real_label.cpu().numpy(), pre_label.cpu().numpy(), predicted_probs.cpu().numpy())
            
            
            # 打印
            print_str = (f"Accuracy: {metrics['Accuracy']}, Precision: {metrics['Precision']}, "
                        f"Recall: {metrics['Recall']}, F1 Score: {metrics['F1 Score']}, AUC: {metrics['AUC']}")
            
            acc2.append(metrics['Accuracy'])
            precision2.append(metrics['Precision'])
            recall2.append(metrics['Recall'])
            f12.append(metrics['F1 Score'])
            auc2.append(metrics['AUC'])
            spec2.append(metrics['Specificity'])
            ssim2.append(np.mean(ssim))
            psrn2.append(np.mean(psrn))
            
            # 使用正则表达式提取数字
            print(print_str)
        
        


    avg_acc2 = sum(acc2) / len(acc2)
    avg_precision2 = sum(precision2) / len(precision2)
    avg_recall2 = sum(recall2) / len(recall2)
    avg_f12 = sum(f12) / len(f12)
    avg_auc2 = sum(auc2) / len(auc2)
    avg_spec2 = sum(spec2) / len(spec2)
    avg_ssim2 = sum(ssim2) / len(ssim2)
    avg_psrn2 = sum(psrn2) / len(psrn2)


    var_acc2 = np.var(acc2,ddof=1)
    var_precision2 = np.var(precision2,ddof=1)
    var_recall2 = np.var(recall2,ddof=1)
    var_f12 = np.var(f12,ddof=1)
    var_auc2 = np.var(auc2,ddof=1)
    var_spec2 = np.var(spec2, ddof=1)
    var_ssim2 = np.var(ssim2, ddof=1)
    var_psrn2 = np.var(psrn2, ddof=1)


    # 4. Save the metrics to a CSV file
    results_df2 = pd.DataFrame({
        'Fold': list(range(1, 6)) + ['Average', 'Variance'],
        'Accuracy': acc2 + [avg_acc2, var_acc2],
        'Precision': precision2 + [avg_precision2, var_precision2],
        'Recall': recall2 + [avg_recall2, var_recall2],
        'F1 Score': f12 + [avg_f12, var_f12],
        'AUC': auc2 + [avg_auc2, var_auc2],
        'Specificity': spec2 + [avg_spec2, var_spec2],
        'SSIM': ssim2 + [avg_ssim2, var_ssim2],
        'PSRN': psrn2 + [avg_psrn2, var_psrn2],
        
    })

    results_df2.to_csv("Test_result04/"+opt2['version']+"04.csv", index=False)
            
