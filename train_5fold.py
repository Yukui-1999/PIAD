from torch.utils.data import  DataLoader,random_split
import torch
from model.MADL import MADL
from data.dataset import CTCTA_dataset_truefalse_new
from sklearn.metrics import classification_report
import numpy as np
import os
import wandb
import SimpleITK as sitk
import random
from sklearn.preprocessing import MinMaxScaler
from util import compute_binary_classification_metrics,import_data,get_train_test_sets,train_cfd,cfd_model
from util import calculate_ssim_psnr,multi_class_dice_score_jaccard_score


seed = 47
random.seed(seed)
os.environ['PYTHONHASHSEED'] =str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic =True

scaler = MinMaxScaler()
data = import_data("/mnt/data/CT_CTA/data_truefalse_vresion2/all","/mnt/data/CT_CTA/cfd_res/physical_infor.csv")

indices = np.arange(data["isad"].shape[0])
np.random.shuffle(indices)
# Split the shuffled indices into 5 folds
fold_indices = np.array_split(indices, 5)


all_metrics = []
train_logs = {}
val_logs = {}
for fold in range(5):
    print(f"Starting Fold {fold + 1}")
    
    version = 'version63_'+str(fold)
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
        "lr_c":0.00001,
        "lr_d":0.0001,
        "lr_cfd":0.00001,
        "norm":'unet:gn,trans:gn,classifier:gn',
        "classification_threshold":0.5,
        "cfd_embedding":True,
        "include_background":False,
        "seed_value" : seed,
        "pretrained_cfd":True,
        "cfd_classifer":False,
        "train_ratio":5 ,
        "traincfd_epoch":80,
        "version":version,
        "pyhsical_dim":3,
        "cross_attention_dim":32,
        "Test":False,
        "Ablation_transformer_use":True,
        "classifier_post":False,
    }
    device = torch.device( opt['device'] if torch.cuda.is_available() else 'cpu')
    trainset_sample, testset_sample = get_train_test_sets(fold, data, fold_indices)
    cfd = scaler.fit_transform(trainset_sample["cfd"])
    trainset_sample["cfd"] = cfd
    train_dataset = CTCTA_dataset_truefalse_new(trainset_sample)
    cfd_test = scaler.transform(testset_sample["cfd"])
    testset_sample["cfd"] = cfd_test
    val_dataset = CTCTA_dataset_truefalse_new(testset_sample)
    # Prepare data loaders
    
    train_dataloader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt['batch_size'])
        
    cfdmodel = None
    if opt['cfd_embedding']:
        if opt["pretrained_cfd"]:
            
            cfdmodel = train_cfd(train_dataloader,val_dataloader,opt)
    else:
        if opt["cfd_classifer"]:
            
            cfdmodel = train_cfd(train_dataloader,val_dataloader,opt)
    
    model = MADL(device=device,opt=opt,cfdmodel=cfdmodel).to(device)
    wandb.init(
        project="final_ctcta",
        name=version,
        config=opt
    )
    root = "/mnt/data/CT_CTA/Final_ctcta/checkpoint"

    checkpoint_dir = os.path.join(root,version)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = [version+'_e10.pth',version+'_e50.pth',version+'_e100.pth']
    checkpoint_path = [os.path.join(checkpoint_dir,filename[0]),os.path.join(checkpoint_dir,filename[1]),os.path.join(checkpoint_dir,filename[2])]
    trained_model_dir = os.path.join("/mnt/data/CT_CTA/Final_ctcta/trained_model",version)
    best_model_dir = os.path.join("/mnt/data/CT_CTA/Final_ctcta/best_model",version)
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)

    trained_model_path = os.path.join(trained_model_dir,"weight.pth")
    best_model_path = os.path.join(best_model_dir,"weight.pth")
    start_epoch = model.load_model_from_checkpoint(checkpoint_path[0])
    
    num_epochs = opt['epoch']
    with open("/home/dingzhengyao/CT_CTA/Final_ctcta/logs/log_"+version +".txt", "a") as log_file:
        log_file.write(version)
        print(opt)
        best_f1 = -float("inf")
        best_recall = -float("inf")
        for epoch in range(start_epoch,num_epochs):
            model.train()
            
            for name, optimizer in model.optimizers.items():
                wandb.log({f"{name}-lr": optimizer.param_groups[0]['lr']})
        
            for batch_idx, (real_images, target_images,label,isad,cfd) in enumerate(train_dataloader):
                
                step = epoch * len(train_dataloader) + batch_idx

                model.set_input(real_images,target_images,label,isad,cfd)
                model.optimize_parameters(batch_idx)
                if batch_idx%10==0:
                    current_train_log = model.log(batchidx=batch_idx,logfile=log_file,wandb=wandb)
                if step not in train_logs:
                    train_logs[step] = []
                train_logs[step].append(current_train_log)
            # 在每个epoch后更新学习率调度器
            for name, scheduler in model.schedulers.items():
                if name == "cfd" and opt["pretrained_cfd"]:
                    continue
                scheduler.step()
            
            print(f"epoch:{epoch}正常运行")
            log_file.write(f"epoch:{epoch}正常运行\n")
            log_file.write("\n\n")
            log_file.flush()

            if (epoch+1) % 10 == 0:
                model.save_model(epoch=epoch,path=checkpoint_path[0])
            if (epoch+1) % 50 == 0:
                model.save_model(epoch=epoch,path=checkpoint_path[1])
            if (epoch+1) % 100 == 0:
                model.save_model(epoch=epoch,path=checkpoint_path[2])
                
            # 验证阶段
            model.eval()
            with torch.no_grad():
               
                ssim = []
                psrn = []
                dice_score = []
                jaccard_score = []
                val_losses_G_Gan = []
                val_losses_G_L1 = []
                val_losses_G_Per = []
                val_losses_G_CE = []
                val_losses_G_seg = []
            

                val_losses_D_fake = []
                val_losses_D_real = []
                val_losses_C = []
                val_losses_cfd = []

                predicted_probs = torch.Tensor([]).float().to(device)
                pre_label = torch.Tensor([]).long().to(device)  # assuming you're using a device object for your model's device
                real_label = torch.Tensor([]).long()
                
                for batch_idx, (real_images, target_images,label,isad,cfd) in enumerate(val_dataloader):
                    # onehot_label = to_one_hot(label,2)
                    # label = label.view(-1, 1).to(torch.float32)
                    model.set_input(real_images,target_images,label,isad,cfd)
                    model.forward()  # compute predictions
                    model.backward_D(compute_gradients=False)
                    model.backward_cfd(compute_gradients=False)
                    model.backward_C(compute_gradients=False)
                    model.backward_G(compute_gradients=False)
                    output_seg = model.generate_segment
                    output_gen = model.generate_cta
                    ssim_,psrn_ = calculate_ssim_psnr(output_gen.cpu(),target_images)
                    dice_,jaccard_ = multi_class_dice_score_jaccard_score(output_seg.cpu(),label)
                    ssim.append(ssim_)
                    psrn.append(psrn_)
                    dice_score.append(dice_)
                    jaccard_score.append(jaccard_)

                    val_losses_G_CE.append(model.loss_G_CE.item())
                
                    val_losses_G_L1.append(model.loss_G_L1.item())
                    val_losses_G_Per.append(model.perceptual_loss.item())
                    val_losses_G_seg.append(model.loss_G_seg.item())
                    val_losses_G_Gan.append(model.loss_G_GAN.item())

                    val_losses_D_fake.append(model.loss_D_fake.item())
                    val_losses_D_real.append(model.loss_D_real.item())
                    val_losses_C.append(model.loss_C.item())
                    val_losses_cfd.append(model.loss_cfd.item())

                    predicted_probs = torch.cat((predicted_probs,torch.sigmoid(model.pred_ad)),dim=0)
                    pre_label = torch.cat((pre_label, (torch.sigmoid(model.pred_ad) > opt['classification_threshold']).float()), dim=0)
                    real_label = torch.cat((real_label, isad), dim=0)
                # 计算验证集上的平均损失
                ssim = np.mean(ssim)
                psrn = np.mean(psrn)
                dice_score = np.mean(dice_score)
                jaccard_score = np.mean(jaccard_score)
                val_losses_G_CE = np.mean(val_losses_G_CE)
                val_losses_G_L1 = np.mean(val_losses_G_L1)
            
                val_losses_G_Per = np.mean(val_losses_G_Per)
                val_losses_G_seg = np.mean(val_losses_G_seg)
                val_losses_G_Gan = np.mean(val_losses_G_Gan)

                val_losses_D_fake = np.mean(val_losses_D_fake)
                val_losses_D_real = np.mean(val_losses_D_real)
                val_losses_C = np.mean(val_losses_C)
                val_losses_cfd = np.mean(val_losses_cfd)

                metrics = compute_binary_classification_metrics(real_label.cpu().numpy(), pre_label.cpu().numpy(), predicted_probs.cpu().numpy())
                
                # 打印
                print_str = (f"Epoch {epoch}, "
                            f"Val Loss G_CE: {val_losses_G_CE}, Val Loss G_L1: {val_losses_G_L1}, Val Loss G_Per: {val_losses_G_Per}, "
                            f"Val Loss G_seg: {val_losses_G_seg}, Val Loss G_Gan: {val_losses_G_Gan}, "
                            f"Val Loss D_fake: {val_losses_D_fake}, Val Loss D_real: {val_losses_D_real}, "
                            f"Val Loss C: {val_losses_C}, Val Loss cfd: {val_losses_cfd}, "
                            f"Accuracy: {metrics['Accuracy']}, Precision: {metrics['Precision']}, "
                            f"Recall: {metrics['Recall']}, F1 Score: {metrics['F1 Score']}, AUC: {metrics['AUC']},Spec:{metrics['Specificity']}")

                print(print_str)
                if metrics['F1 Score'] > best_f1 or (metrics['F1 Score'] == best_f1 and metrics['Recall'] > best_recall):
                    if metrics['F1 Score'] > best_f1:
                        best_f1 = metrics['F1 Score']
                    if metrics['Recall'] > best_recall:
                        best_recall = metrics['Recall']
                    torch.save(model.state_dict(), best_model_path)
                    message = (f"New best model saved with F1 Score: {best_f1} and Recall: {best_recall}")
                    print(message)
                    # wandb.log({"Best Model Message": message})
                current_val_log = {
                    "Val Loss G_CE": val_losses_G_CE,
                    "Val Loss G_L1": val_losses_G_L1,
                    "Val Loss G_Per": val_losses_G_Per,
                    "Val Loss G_seg": val_losses_G_seg,
                    "Val Loss G_Gan": val_losses_G_Gan,
                    "Val Loss D_fake": val_losses_D_fake,
                    "Val Loss D_real": val_losses_D_real,
                    "Val Loss C": val_losses_C,
                    "Val Loss cfd": val_losses_cfd,
                    "Accuracy": metrics['Accuracy'],
                    "Precision": metrics['Precision'],
                    "Recall": metrics['Recall'],
                    "F1 Score": metrics['F1 Score'],
                    "AUC": metrics['AUC'],
                    "Spec":metrics['Specificity'],
                    "ssim":ssim,
                    "psrn":psrn,
                    "DSC":dice_score,
                    "Jacacard":jaccard_score
                }
                step = epoch 
                if step not in val_logs:
                    val_logs[step] = []
                val_logs[step].append(current_val_log)
                # Logging for wandb
                wandb.log(current_val_log)



        model_save_path = trained_model_path
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        log_file.write(f"Model saved to {model_save_path}\n")
        wandb.finish()



    # test
    root = '/mnt/data/CT_CTA/data_truefalse_vresion2/test'
    dirlist = os.listdir(root)
    generatepath = os.path.join("/mnt/data/CT_CTA/Final_ctcta/generate",version)
    if not os.path.exists((generatepath)):
        os.mkdir(generatepath)
    for i in dirlist:
        path = os.path.join(root,i)
        ct_path = os.path.join(path,'ct.nii.gz')
        cta_path = os.path.join(path,'cta.nii.gz')

        true_path = os.path.join(path,'true.nii.gz')
        false_path = os.path.join(path,'false.nii.gz')
        
        label_true_array = sitk.GetArrayFromImage(sitk.ReadImage(true_path))
        label_false_array = sitk.GetArrayFromImage(sitk.ReadImage(false_path))
        merge_label = label_true_array + label_false_array * 2
        cta_array = sitk.GetArrayFromImage(sitk.ReadImage(cta_path))

        generate_seg = os.path.join(generatepath,i+'_seg')+'.nii.gz'
        generate = os.path.join(generatepath,i)+'.nii.gz'

        ct_image=sitk.ReadImage(ct_path)
        ct_array=sitk.GetArrayFromImage(ct_image)
        showct_array = ct_array
        ct_array = np.expand_dims(ct_array,axis=0)
        ct_array = np.expand_dims(ct_array,axis=0)
        
        model.eval()
        print(device)
        input_data = torch.from_numpy(ct_array)
        input_data = input_data.to(device)

        # 通过生成器生成新的图像
        with torch.no_grad():
            cfd = model.cfdpredict(input_data)
            generate_segment ,generate_cta ,encoder_output = model.generator(input_data,cfd)

        output_data=generate_cta.cpu()
        output_data_seg=generate_segment.cpu()


        print(output_data.shape)
        generate_array = output_data[0][0]
        generate_array_seg = output_data_seg[0]
        result = np.concatenate((showct_array,cta_array,generate_array), axis=1)#对比展示
        
        image=sitk.GetImageFromArray(result)
        image.SetSpacing((1.3, 1.3, 5))
        sitk.WriteImage(image, generate)

        segmentation = torch.argmax(generate_array_seg, axis=0)
        segmentation_np = segmentation.cpu().numpy()
        result = np.concatenate((segmentation_np,merge_label), axis=1)#对比展示
        sitk_image = sitk.GetImageFromArray(result.astype(np.int16))
        sitk_image.SetSpacing((1.3, 1.3, 5))
        sitk.WriteImage(sitk_image, generate_seg)

name = "5fold_avg" + version
wandb.init(
        project="final_ctcta",
        name=name,
        config=opt
    )

print(val_logs)
average_train_logs = {}
for step, logs in train_logs.items():
    average_log = {}
    for key in logs[0].keys():
        average_log[key] = sum([log[key] for log in logs]) / len(logs)
    average_train_logs[step] = average_log

average_val_logs = {}
for step, logs in val_logs.items():
    average_log = {}
    for key in logs[0].keys():
        average_log[key] = sum([log[key] for log in logs]) / len(logs)
    average_val_logs[step] = average_log

for step, log in average_train_logs.items():
    wandb.log(log, step=step)
print("???//??/")
print(average_val_logs)
for step, log in average_val_logs.items():
    wandb.log(log)


wandb.finish()



