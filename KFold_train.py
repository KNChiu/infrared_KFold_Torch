#%%
from functools import total_ordering
import os
import cv2
import random

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix ,roc_curve, auc

import matplotlib.pyplot as plt 
from itertools import cycle
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from model.patch_convmix_convnext import PatchConvmixConvnext
from model.focal_loss import FocalLoss
import json

import pandas as pd
import seaborn as sns

import wandb
import time

#%%   
def gradCamImg(model, logPath):
    # checkpoint = r'save.pt'
    image_path = r'data\new_cc\correct\0a3d5f0d-6d26-4186-a4fe-7d6b2869216e_20210511_105912.png'
    
    # img = cv2.imread(image_path)
    # img = cv2.resize(img, (640, 640))

    # torch.save(model.state_dict(), "save.pt")
    
    rgb_img, cam_image, cam_mask, cam_gb, gb = Grad_CAM.get_cam(model=model, image_path=image_path)


    fig = plt.figure()
    subplot1=fig.add_subplot(2, 2, 1)
    subplot1.imshow(rgb_img)

    subplot2=fig.add_subplot(2, 2, 2)
    subplot2.imshow(cam_image)

    subplot3=fig.add_subplot(2, 2, 3)
    subplot3.imshow(cam_mask)

    subplot4=fig.add_subplot(2, 2, 4)
    subplot4.imshow(cam_gb)

    fig.suptitle("Grad Cam of Training")

    plt.savefig(logPath+"//"+"gradCam_"+str(time.strftime("%H%M%S", time.localtime()))+".jpg", bbox_inches='tight')
    plt.close('all')

def confusion(y_true, y_pred, calsses, logPath=None):
    confmat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center', fontsize=10)

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)

    if (TN + FP) == 0:
        Specificity = 0.0
    else:
        Specificity = TN / (TN + FP)

    if (TP + FN) == 0:
        Sensitivity = 0.0
    else:
        Sensitivity = TP / (TP + FN)
    
    if logPath:
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix, calsses, calsses)
        plt.figure(figsize = (9,6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        plt.xlabel('Predict', fontsize=10)        
        plt.ylabel('True', fontsize=10)
        
        plt.title('Accuracy : {:.2f} | Specificity : {:.2f} | Sensitivity : {:.2f}'.format(Accuracy, Specificity, Sensitivity), fontsize=10)
        plt.savefig(logPath + "//" + "confusion.jpg", bbox_inches='tight')
        # plt.close('all')
        plt.close()

    return Accuracy, Specificity, Sensitivity

def compute_auc(y_true, y_score, classes, logPath=None):
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

    fpr, tpr, roc_auc = dict(), dict(), dict()
    y_label = np.eye(len(classes))[y_true]
    y_label = y_label.reshape(len(y_true), len(classes))
    y_score = np.array(y_score)

    if len(y_label) > 1 :
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i], pos_label=1)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        if logPath:
            plt.figure()
            
            for i, color in zip(range(len(classes)), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw = 2,
                        label='ROC curve of level {0} (area = {1:0.2f})'
                        ''.format(i + 1, roc_auc[i]))
            
            plt.plot([0, 1], [0, 1], 'k--', lw = 2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('multi-calss ROC')
            plt.legend(loc="lower right")
            plt.savefig(logPath + "//" + "compute_auc.jpg", bbox_inches='tight')
            plt.close()
    else:
        roc_auc = -1.00

    return roc_auc

def fit_model(model, train_loader, val_loader, classes):
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    loss_func = FocalLoss(class_num=3, alpha = torch.tensor([0.3, 0.3, 0.3]).to(device), gamma = 4)
    # loss_func = FocalLoss(class_num=3, alpha = None, gamma = 4)

    for epoch in range(EPOCH):
        model.train()
        training_loss = 0

        for idx, (x, y) in enumerate(train_loader):
            output = model(x.to(device))
            outPred = output[0]

            loss = loss_func(outPred, y.to(device))
            training_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        y_pred_score =[]
        y_true = []
        with torch.no_grad():
            val_loss = 0
            for idx, (x_, y_) in enumerate(val_loader):
                pred = model(x_.to(device))
                loss_ = loss_func(pred, y_.to(device))
                val_loss += loss_.item()

                if type(pred) == tuple:
                    pred = pred[0]

                y_pred_score += pred.tolist()
                y_true += y_.tolist()
                
            roc_auc = compute_auc(y_true, y_pred_score, classes)

        if roc_auc != -1:
            roc_auc = max(roc_auc.values())

        if WANDBRUN:
            wb_run.log({ 
                        "Epoch" : epoch + 1,
                        "Training Loss": training_loss,
                        "Val Loss": val_loss,
                        "Val AUC" : roc_auc,
                    })
            # wb_run.finish()
            
        print('  => Epoch : {}  Training Loss : {:.4e}  Val Loss : {:.4e}  Val AUC : {:.2}'.format(epoch + 1, training_loss, val_loss, roc_auc))

        # torch.save(model.state_dict(), "last.pth")
    return training_loss, val_loss

def test_model(model, test_loader, classes):
    model.eval()
    correct = 0
    y_true = []
    y_pred = []
    y_pred_score = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            pred = model(x.to(device))

            if type(pred) == tuple:
                pred = pred[0]

            y_pred_score += pred.tolist()

            pred = torch.max(pred.data, 1)[1] 
            correct += (pred == y.to(device)).sum()
            # print('Pred : {}, True : {}'.format(pred, y))

            y_true += y.tolist()
            y_pred += pred.tolist()
            
        
        Accuracy, Specificity, Sensitivity = confusion(y_true, y_pred, classes)
        roc_auc = compute_auc(y_true, y_pred_score, classes)

        return Accuracy, roc_auc, Specificity, Sensitivity, y_true, y_pred, y_pred_score

if __name__ == '__main__':
    ISKFOLD = True
    KFOLD_N = 3
    SAVEPTH = False
    SAVEIDX = True
    WANDBRUN = True
    SEED = 42
    
    CLASSNANE = ['Ischemia', 'Acutephase', 'Recoveryperiod']

    EPOCH = 100
    BATCHSIZE = 16
    LR = 0.0001

    DATAPATH = r'C:\Data\外科溫度\裁切\已分訓練集\cut_3_kfold'

    if SEED:
        '''設定隨機種子碼'''
        os.environ["PL_GLOBAL_SEED"] = str(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    logPath = r"logs//" + str(time.strftime("%m%d_%H%M", time.localtime()))
    if not os.path.isdir(logPath):
        os.mkdir(logPath)

    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = PatchConvmixConvnext(dim = 768, depth = 3, kernel_size = 7, patch_size = 16, n_classes = 3).to(device)
    # if WANDBRUN:
    #     wandb.watch(model)

    transform = transforms.Compose([transforms.Resize((640, 640)),
                                    transforms.ToTensor()])

    if ISKFOLD:
        dataset = ImageFolder(DATAPATH, transform)

        kf = KFold(n_splits = KFOLD_N, shuffle = True)
        Kfold_cnt = 0
        acc_array = []
        total_true = []
        total_pred = []
        total_pred_score = []
        totlal_acc = 0

        # json_dict = {'Kfold_cnt' :[], 'train_idx': [], 'val_idx': []}
        # json_file = open(logPath + '//'+ 'kfold_idx.json', "w")
        # json.dump(json_dict, json_file, indent=4)
        # json_file.close()

        for train_idx, val_idx in kf.split(dataset):
            if WANDBRUN:
                wb_run = wandb.init(project='infraredThermal_kfold', entity='y9760210', reinit=True, group="KFold_class_3", name=str("kfold_N="+str(Kfold_cnt+1)))
            Kfold_cnt += 1

            if SAVEIDX:
                json_dict = {'Kfold_cnt' : Kfold_cnt, 'val_idx': val_idx.tolist()}
                json_file = open(logPath + '//'+ 'kfold_idx.json', "a")
                json.dump(json_dict, json_file, indent=4)
                json_file.close()


            # if not os.path.isdir(KFoldPath):
            #     os.mkdir(KFoldPath)
            # print(KFoldPath)
            
            train = Subset(dataset, train_idx)
            val = Subset(dataset, val_idx)
        
            train_loader = DataLoader(train, batch_size = BATCHSIZE, shuffle = True, num_workers = 2)
            val_loader = DataLoader(val, batch_size = BATCHSIZE, shuffle = True, num_workers = 2)
            
            fit_model(model, train_loader, val_loader, CLASSNANE)

            Accuracy, roc_auc, Specificity, Sensitivity, kfold_true, kfold_pred,  kfold_pred_score = test_model(model, val_loader, CLASSNANE)
            totlal_acc += Accuracy
            total_true += kfold_true
            total_pred += kfold_pred
            total_pred_score += kfold_pred_score
            if roc_auc != -1:
                roc_auc = max(roc_auc.values())

            if WANDBRUN:
                wb_run.log({
                            "Accuracy" : Accuracy,
                            "Test AUC" : roc_auc,
                            "Test Specificity" : Specificity,
                            "Test Sensitivity" : Sensitivity
                        })
                # wb_run.finish()

            print("===================================================================================================")
            print('Kfold : {} , Accuracy : {:.2e} , Test AUC : {:.2} , Specificity : {:.2} , Sensitivity : {:.2}'.format(Kfold_cnt, Accuracy, roc_auc, Specificity, Sensitivity))
            print("===================================================================================================")
            
            if SAVEPTH:
                saveModelpath = logPath + "//" + str(Kfold_cnt) + "_last.pth"
                torch.save(model.state_dict(), saveModelpath)

        # Kfold 結束交叉驗證
        Accuracy, Specificity, Sensitivity = confusion(total_true, total_pred, CLASSNANE, logPath)
        roc_auc = compute_auc(total_true, total_pred_score, CLASSNANE, logPath)
        print("===================================================================================================")
        print('Kfold:N= : {} ,Total = Accuracy : {:.3} , Specificity : {:.2} , Sensitivity : {:.2}'.format(KFOLD_N, Accuracy, Specificity, Sensitivity))
        print("Total AUC: ", roc_auc)
        print("===================================================================================================")
        if roc_auc != -1:
            roc_auc = max(roc_auc.values())
        if WANDBRUN:
            wb_run.log({
                        "KFold Accuracy" : Accuracy,
                        "KFold AUC" : roc_auc,
                        "KFold Specificity" : Specificity,
                        "KFold Sensitivity" : Sensitivity
                    })
            wb_run.finish()

    if WANDBRUN:
        wb_run.finish()

