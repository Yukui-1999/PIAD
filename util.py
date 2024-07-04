from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,confusion_matrix
from torch.utils.data import  DataLoader,random_split
import torch
import pandas as pd
from model.MADL import MADL
from data.dataset import CTCTA_dataset_truefalse
import numpy as np
import torch.nn.functional as F
from skimage import metrics
import numpy as np
import os
import wandb
import SimpleITK as sitk
from model.TransUnet_cfd.TransBTS_downsample8x_skipconnection import TransBTS_cfd
import wandb
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim

def calculate_ssim_psnr(output, target):
    """
    Calculate SSIM and PSNR for a batch of images.

    :param output: Model output, shape (batch, 1, depth, height, width), values in range [-1, 1]
    :param target: Ground truth, shape (batch, 1, depth, height, width), values in range [-1, 1]
    :return: Average SSIM and PSNR for the batch
    """
    # Scale and shift the images from [-1, 1] to [0, 255]
    output = ((output + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
    target = ((target + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)

    batch_size = output.shape[0]
    ssim_total = 0
    psnr_total = 0
    count = 0  # 用于记录实际计算了多少对切片

    for i in range(batch_size):
        for j in range(output.shape[2]):
            output_img = output[i, 0, j]
            target_img = target[i, 0, j]
            
            if not np.array_equal(output_img, target_img):
                ssim_total += metrics.structural_similarity(output_img, target_img, data_range=255)
                psnr_total += metrics.peak_signal_noise_ratio(output_img, target_img, data_range=255)
                count += 1  # 增加计数
            else:
                print(f"跳过完全相同的切片:batch {i}, slice {j}")

    # 只有当至少计算了一对切片时才计算平均值
    if count > 0:
        average_ssim = ssim_total / count
        average_psnr = psnr_total / count
    else:
        average_ssim = None
        average_psnr = None
        print("没有计算任何切片的SSIM或PSNR")
    
    return average_ssim, average_psnr
def multi_class_dice_score_jaccard_score(output, target, epsilon=1e-6):
    """
    Calculate multi-class Dice Score for a batch of images.

    :param output: Model output, shape (batch, num_classes, depth, height, width)
    :param target: Ground truth, shape (batch, 1, depth, height, width)
    :param epsilon: Small constant to avoid division by zero
    :return: Multi-class Dice Score
    """
    # Apply softmax to the model output
    output = F.softmax(output, dim=1)
    
    # Remove the singleton dimension and convert the ground truth to long type
    target = target.squeeze(1).long()
    
    # Convert the ground truth to one-hot encoding
    target_one_hot = F.one_hot(target, num_classes=output.shape[1]).float()
    target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3)
    
    # Calculate TP, FP, and FN
    tp = (output * target_one_hot).sum(dim=(0, 2, 3, 4))
    fp = (output * (1 - target_one_hot)).sum(dim=(0, 2, 3, 4))
    fn = ((1 - output) * target_one_hot).sum(dim=(0, 2, 3, 4))
    
    dice_score = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
    jaccard_score = (tp + epsilon) / (tp + fp + fn + epsilon)
    
    average_dice_score = dice_score.mean()
    average_jaccard_score = jaccard_score.mean()
    
    return average_dice_score.item(),average_jaccard_score.item()


def compute_binary_classification_metrics(real_labels, predicted_labels, predicted_probs):
    print(f'real_labels.shape:{real_labels.shape}')
    print(f'predicted_labels.shape:{predicted_labels.shape}')
    real_labels = real_labels.squeeze().astype(int)
    predicted_labels = predicted_labels.squeeze().astype(int)
    predicted_probs = predicted_probs.squeeze().astype(float)
    
    # Calculate basic metrics
    accuracy = accuracy_score(real_labels, predicted_labels)
    precision = precision_score(real_labels, predicted_labels)
    recall = recall_score(real_labels, predicted_labels)
    f1 = f1_score(real_labels, predicted_labels)

    # For ROC and AUC
    fpr, tpr, _ = roc_curve(real_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(real_labels, predicted_labels).ravel()
    specificity = tn / (tn + fp)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": roc_auc,
        "Specificity": specificity
    }

def import_data(imagepath,cfdpath):
    data_path = imagepath
    cfd_path = cfdpath
    print(f"data_path:{data_path},cfd_path:{cfdpath}")
    
    data_dir = os.listdir(data_path)
    ct_array=[]
    cta_array=[]
    label_true=[]
    label_false=[]

    isad=[]
    oa=[]
    v_minus=[]
    wbeta=[]

    cfdres = pd.read_csv(cfd_path)
    def get_is_ad_value(name):
        return cfdres[cfdres['name'] == name]['isad_v3'].values[0]
    def get_v_minus(name):
        return cfdres[cfdres['name'] == name]['v_minus'].values[0]
    def get_oa(name):
        return cfdres[cfdres['name'] == name][' Orthogonality Angle [ degree ]'].values[0]
    def get_wbeta(name):
        return cfdres[cfdres['name'] == name][' Velocity w.Beta'].values[0]
    for index in data_dir:
        # if index in adlist and index not in xzlist:
        # print(index)
        index_path = os.path.join(data_path,index)
        ct_path = os.path.join(index_path,'ct.nii.gz')
        cta_path = os.path.join(index_path,'cta.nii.gz')
        label_true_path = os.path.join(index_path,'true.nii.gz')
        label_false_path = os.path.join(index_path,'false.nii.gz')

        ct_array_s = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
        cta_array_s = sitk.GetArrayFromImage(sitk.ReadImage(cta_path))
        label_true_array_s = sitk.GetArrayFromImage(sitk.ReadImage(label_true_path))
        label_false_array_s = sitk.GetArrayFromImage(sitk.ReadImage(label_false_path))

        ct_array.append(ct_array_s)
        cta_array.append(cta_array_s)
        label_true.append(label_true_array_s)
        label_false.append(label_false_array_s)

        
        isad.append(get_is_ad_value(int(index)))
        oa.append(get_oa(int(index)))
        v_minus.append(get_v_minus(int(index)))
        wbeta.append(get_wbeta(int(index)))

    ct_array = np.array(ct_array)
    cta_array = np.array(cta_array)
    label_true = np.array(label_true)
    label_false = np.array(label_false)

    isad = np.array(isad)
    oa = np.array(oa)
    v_minus = np.array(v_minus)
    wbeta = np.array(wbeta)
    cfd = np.column_stack((v_minus,oa,wbeta))
    
    # print(ct_array.shape)

    ct_array = np.expand_dims(ct_array,axis=1)
    cta_array = np.expand_dims(cta_array,axis=1)
    label_true = np.expand_dims(label_true,axis=1)
    label_false = np.expand_dims(label_false,axis=1)
    merge_label = label_true + label_false*2
    # print(merge_label.min(),merge_label.max())
    # print(merge_label.shape)
    # print(f"ct_array.shape:  {ct_array.shape},ct_array.min:{ct_array.min()},ct_array.max:{ct_array.max()}")
    # print(f"cta_array.shape:  {cta_array.shape},cta_array.min:{cta_array.min()},cta_array.max:{cta_array.max()}")
    # print(f"label_true.shape:  {label_true.shape}")
    # print(f"label_false.shape:  {label_false.shape}")
    # print(f"is_ad:{isad}")
    # print(f"cfd.shape:  {cfd.shape}")

    data={"ct_array":ct_array,
                "cta_array":cta_array,
                "merge_label":merge_label,
                "isad":isad,
                "cfd":cfd
                }
    return data


def get_train_test_sets(fold_num, data_dict, fold_indices):
    """
    Given a fold number (from 0 to 4), a data dictionary, and fold indices, 
    returns two dictionaries: trainset and testset.
    
    Parameters:
    - fold_num: An integer between 0 and 4.
    - data_dict: Dictionary containing the data arrays.
    - fold_indices: List of index arrays for each fold.
    
    Returns:
    Two dictionaries: trainset and testset.
    """
    # Validate the fold number
    if fold_num < 0 or fold_num > 4:
        raise ValueError("Fold number must be between 0 and 4.")
    
    # Set the test set based on the fold number
    test_indices = fold_indices[fold_num]
    
    # All other folds are the training set
    train_indices = np.concatenate([fold_indices[i] for i in range(5) if i != fold_num])
    
    # Extract the training and test sets for each array in the data_dict
    trainset = {}
    testset = {}
    for key, array in data_dict.items():
        trainset[key] = array[train_indices]
        testset[key] = array[test_indices]
    
    return trainset, testset
def cfd_model(opt,fold):
    path = ['version49_0','version49_1','version49_2','version49_3','version49_4']
    trained_model_dir = os.path.join("/mnt/data/CT_CTA/Final_ctcta/trained_model",path[fold])
    trained_model_path = os.path.join(trained_model_dir,"weight_cfd.pth")
    if opt['seed_value']==42:
        checkpoint = torch.load(trained_model_path)
        _ , best_model_instance = TransBTS_cfd()
        best_model_instance.load_state_dict(checkpoint)
        best_model_instance = best_model_instance.to(opt["device"])
        return best_model_instance
def train_cfd(train_dataloader,val_dataloader,opt):
    name = opt["version"]+"train_cfd"
    
    best_val_loss = float('inf')  # Initialize with a high value
    
    _ , model = TransBTS_cfd()
    model = model.to(opt["device"])
    trained_model_dir = os.path.join("/mnt/data/CT_CTA/Final_ctcta/trained_model",opt["version"])
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)
    trained_model_path = os.path.join(trained_model_dir,"weight_cfd.pth")
    if os.path.exists(trained_model_path):
        checkpoint = torch.load(trained_model_path)
        _ , best_model_instance = TransBTS_cfd()
        best_model_instance.load_state_dict(checkpoint)
        best_model_instance = best_model_instance.to(opt["device"])
        return best_model_instance
    
    wandb.init(
        project="final_ctcta",
        name=name,
        config=opt
    )
    optimizer = optim.AdamW(model.parameters(), lr=opt['lr_cfd'], betas=(0.5, 0.999))
    midepoch = (opt["traincfd_epoch"]+1) / 2
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - (epoch - midepoch) / midepoch if epoch >= midepoch else 1)
    criterionL1 = torch.nn.L1Loss()
    for epoch in range(0,opt["traincfd_epoch"]):
        model.train()
        for batch_idx, (real_images, target_images,label,isad,cfd) in enumerate(train_dataloader):
            real_images = real_images.to(opt["device"])
            cfd = cfd.to(opt["device"])

            output = model(real_images)
            loss = criterionL1(output,cfd)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % 10==0:
                print(f"traincfd_loss : {loss}")
            wandb.log({"traincfd_loss":loss})
        model.eval()
        with torch.no_grad():
            val_loss_list = []
            for batch_idx, (real_images, target_images,label,isad,cfd) in enumerate(val_dataloader):
                real_images = real_images.to(opt["device"])
                cfd = cfd.to(opt["device"])


                val_output = model(real_images)
                val_loss = criterionL1(val_output,cfd)
                if batch_idx % 10==0:
                    print(f"traincfd_val_loss : {val_loss}")
                wandb.log({"traincfd_val_loss":val_loss})
                val_loss_list.append(val_loss.item())
            val_loss_avg  = np.mean(val_loss_list)
            wandb.log({"traincfd_val_loss_avg":val_loss_avg})
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_model_state = model.state_dict()  # Save the state dict, not the model itself
                torch.save(best_model_state, trained_model_path)
                
                print(f"New best model saved to {trained_model_path} with validation loss {best_val_loss:.4f}\n")

    _ , best_model_instance = TransBTS_cfd()
    best_model_instance.load_state_dict(best_model_state)
    best_model_instance = best_model_instance.to(opt["device"])
    
    wandb.finish()
    return best_model_instance
