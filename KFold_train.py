#%%
from functools import total_ordering
import os
from unittest.mock import patch
import cv2
import random
import io

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix ,roc_curve, auc, accuracy_score

import matplotlib.pyplot as plt 
from itertools import cycle
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

# from model.patch_convmix_convnext import PatchConvmixConvnext
# from model.patch_RepLKNet_DRSN import PatchRepLKNetDRSN
from model.patch_convmix_Attention import PatchConvMixerAttention

from model.load_dataset import MyDataset

from model.focal_loss import FocalLoss
import json

import pandas as pd
import seaborn as sns

import wandb
import time

import catboost as cb

from model.load_dataset import MyDataset
from model.assessment_tool import MyEstimator


def fit_model(model, train_loader, val_loader, classes):
    # optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    optimizer = torch.optim.SGD(model.parameters(), lr = LR)
    # loss_func = FocalLoss(class_num=3, alpha = torch.tensor([0.36, 0.56, 0.72]).to(device), gamma = 4)
    loss_func = FocalLoss(class_num=len(classes), alpha = None, gamma = 4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)    # (1 + T_mult + T_mult**2) * T_0 // 5,15,35,75,155
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=0)
    mini_val_loss = 100
    for epoch in range(EPOCH):
        model.train()
        training_loss = 0

        train_y_pred_score =[]
        train_y_true = []
        train_y_pred = []

        for idx, (x, y, _) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(x.to(device))
            outPred = output[0]

            loss = loss_func(outPred, y.to(device))
            training_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            train_y_pred_score += outPred.tolist()

            # 計算是否正確
            pred = torch.max(outPred.data, 1)[1] 

            train_y_true += y.tolist()
            train_y_pred += pred.tolist()

        scheduler.step()        # cos退火
        cur_lr = optimizer.param_groups[-1]['lr']  

        train_roc_auc, _ = MyEstimator.compute_auc(train_y_true, train_y_pred_score, classes)
        train_Accuracy, Specificity, Sensitivity, _ ,_ = MyEstimator.confusion(train_y_true, train_y_pred, val_keyLabel=None, logPath = None, classes = classes)
        
        # val
        model.eval()
        y_pred_score =[]
        y_true = []
        y_pred = []
        with torch.no_grad():
            val_loss = 0
            for idx, (x_, y_, _) in enumerate(val_loader):
                pred, gap, test_feature = model(x_.to(device))
                loss_ = loss_func(pred, y_.to(device))
                val_loss += loss_.item()

                y_pred_score += pred.tolist()
                
                # 計算是否正確
                pred = torch.max(pred.data, 1)[1] 
                
                y_pred += pred.tolist()
                y_true += y_.tolist()

            roc_auc, _ = MyEstimator.compute_auc(y_true, y_pred_score, classes)
            Accuracy, Specificity, Sensitivity, _, _ = MyEstimator.confusion(y_true, y_pred, val_keyLabel=None, logPath = None, classes = classes)

            
        if roc_auc != -1:
            train_roc_auc = max(train_roc_auc.values())
            roc_auc = max(roc_auc.values())

        if WANDBRUN:
            wb_run.log({ 
                        "Epoch" : epoch + 1,
                        "Training Loss": training_loss,
                        "Train ACC" : train_Accuracy,
                        "Train AUC" : train_roc_auc,
                        "Val Loss": val_loss,
                        "Val AUC" : roc_auc,
                        "Val ACC" : Accuracy,
                        "LR" : cur_lr,
                                 # 將可視化上傳 wandb
                    })

            if epoch % DRAWIMG == 0:    
                if len(gap) > 8:
                    cnt = 8
                else:
                    cnt = len(gap)

                plt.close('all')
                for i in range(cnt):
                    plt.subplot(1, cnt, i+1)
                    plt.imshow(gap[i].cpu().detach().numpy())   # 將注意力圖像取出
                    plt.axis('off')         # 關閉邊框
                
                # plt.show()
                plot_img_np = MyEstimator.get_img_from_fig(plt)    # plt 轉為 numpy
                plt.close('all')
                
                wb_run.log({"val image": [wandb.Image(plot_img_np)]})   # 將可視化上傳 wandb

        if SAVEPTH:
            if SAVEBAST and epoch > 10:
                if mini_val_loss > val_loss:
                    mini_val_loss = val_loss
                    saveModelpath = logPath + "//" + str(Kfold_cnt) + "_bast.pth"
                    torch.save(model.state_dict(), saveModelpath)
            if epoch == EPOCH - 1:
                saveModelpath = logPath + "//" + str(Kfold_cnt) + "_last.pth"
                torch.save(model.state_dict(), saveModelpath)

        print('  => Epoch : {}  Training Loss : {:.4e}  Val Loss : {:.4e}  Val ACC : {:.2}  Val AUC : {:.2}'.format(epoch + 1, training_loss, val_loss, Accuracy, roc_auc))

    return training_loss, val_loss 

def test_model(model, test_loader, classes):
    # test
    if SAVEBAST: 
        saveModelpath = logPath + "//" + str(Kfold_cnt) + "_bast.pth"
    else:
        saveModelpath = logPath + "//" + str(Kfold_cnt) + "_last.pth"

    model.eval()
    model.load_state_dict(torch.load(saveModelpath))
    model.to(device)
    # correct = 0
    y_true = []
    y_pred = []
    y_pred_score = []
    keyLabel = []

    with torch.no_grad():
        for idx, (x, y, key) in enumerate(test_loader):
            keyLabel += key
            pred = model(x.to(device))
            pred, gap, test_feature = pred

            y_pred_score += pred.tolist()

            # 計算是否正確
            pred = torch.max(pred.data, 1)[1] 
            # correct += (pred == y.to(device)).sum()

            y_true += y.tolist()
            y_pred += pred.tolist()
            
        
        Accuracy, Specificity, Sensitivity, error_list, _ = MyEstimator.confusion(y_true, y_pred, val_keyLabel=keyLabel, logPath = None, classes = classes)
        roc_auc, _ = MyEstimator.compute_auc(y_true, y_pred_score, classes)

        return Accuracy, roc_auc, Specificity, Sensitivity, y_true, y_pred, y_pred_score, gap, keyLabel, error_list

def load_feature(dataloader, model):
    feature, label = [], []
    for idx, (x, y, _) in enumerate(dataloader):
        _, _, featureOut = model(x.to(device))

        featureOut = featureOut[0].to('cpu').detach().numpy()
        featureOut = featureOut.reshape((1, -1))[0]

        feature.append(featureOut)
        label.append(y.to('cpu').detach().numpy())
        
    feature = np.array(feature)
    label = np.array(label)
    return feature, label

def catboots_fit(train_data, train_label, val_data, val_label, iterations, CatBoost_depth):
    cbc = cb.CatBoostClassifier(random_state=42, use_best_model=True, iterations=iterations, depth = CatBoost_depth)
    cbc.fit(train_data, train_label,
            eval_set = [(val_data, val_label)],
            verbose=False,
            plot=False
            )
    predict = cbc.predict(val_data)
    predict_Probability = cbc.predict(val_data, prediction_type='Probability')
    return predict, predict_Probability

if __name__ == '__main__':
    ISKFOLD = True
    SAVEPTH = True
    SAVEIDX = True
    WANDBRUN = True
    SAVEBAST = False
    RUNML = False
    SEED = 42
    
    CLASSNANE = ['Ischemia', 'Infect']
    # CLASSNANE = ['Ischemia', 'Acutephase', 'Recoveryperiod']
    CNN_DETPH = 3
    KERNELSIZE = 7


    KFOLD_N = 10
    EPOCH = 155
    BATCHSIZE = 16
    LR = 0.01
    # LR = 0.0001
    DRAWIMG = 20

    CATBOOTS_INTER = 200
    ACTBOOTS_DETPH = 1

    LOGPATH = r'C:\Data\surgical_temperature\trainingLogs\\'
    DATAPATH = r'C:\Data\surgical_temperature\cut\classification\cut_96\\'
    # DATAPATH = r'C:\Data\外科溫度\裁切\已分訓練集\cut_3_kfold'
    # DATAPATH = r'C:\Data\胸大肌\data\3classes\CC\train'

    

    MyEstimator = MyEstimator()
    D = MyDataset(DATAPATH, LOGPATH, 2)


    if SEED:
        '''設定隨機種子碼'''
        os.environ["PL_GLOBAL_SEED"] = str(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # 建立 log
    logPath = LOGPATH + "//logs//" + str(time.strftime("%m%d_%H%M", time.localtime()))
    if not os.path.isdir(logPath):
        os.mkdir(logPath)
        os.mkdir(logPath+'//img//')
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([transforms.Resize((640, 640)),
                                    transforms.ToTensor()])

    if ISKFOLD:
        # dataset = ImageFolder(DATAPATH, transform)          # 輸入數據集
        dataset  = D
        kf = KFold(n_splits = KFOLD_N, shuffle = True)
        Kfold_cnt = 0
        # acc_array = []
        # totlal_acc = 0
        total_true = []
        total_pred = []
        total_pred_score = []

        ML_total_true = []
        ML_total_pred = []
        ML_total_pred_score = []
        total_keyLabel = []

        # KFOLD
        for train_idx, val_idx in kf.split(dataset):
            Kfold_cnt += 1

            if WANDBRUN:
                wb_run = wandb.init(project='infraredThermal_kfold', entity='y9760210', reinit=True, group="KFold_2", name=str("kfold_N="+str(Kfold_cnt)))
            
            if SAVEIDX:
                with open(logPath + '//'+ 'kfold_idx.json','a+',encoding="utf-8") as json_file:
                    json_file.seek(0)  
                    if json_file.read() =='':  
                        data = {}
                    else:
                        json_file.seek(0)
                        data = json.load(json_file)

                    data['Kfold_cnt' + str(Kfold_cnt)] = {'train_idx':train_idx.tolist(), 'val_idx':val_idx.tolist()}

                    json_file.seek(0)
                    json_file.truncate()
                    json.dump(data, json_file, indent=2, ensure_ascii=False)
            
            # 重組 kfold 數據集
            train = Subset(dataset, train_idx)
            val = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train, batch_size = BATCHSIZE, shuffle = True)
            val_loader = DataLoader(val, batch_size = BATCHSIZE, shuffle = True)

            # 匯入模型
            model = PatchConvMixerAttention(dim = 768, depth = CNN_DETPH, kernel_size = KERNELSIZE, patch_size = 16, n_classes = len(CLASSNANE)).to(device)

            # Train
            fit_model(model, train_loader, val_loader, CLASSNANE)

            # Test
            Accuracy, roc_auc, Specificity, Sensitivity, kfold_true, kfold_pred, kfold_pred_score, gap, val_keyLabel, error_list = test_model(model, val_loader, CLASSNANE)

            total_true += kfold_true
            total_pred += kfold_pred
            total_pred_score += kfold_pred_score
            total_keyLabel += val_keyLabel

            if roc_auc != -1:
                roc_auc = max(roc_auc.values())

            print("==================================== CNN Training=================================================")
            print('Kfold : {} , Accuracy : {:.2e} , Test AUC : {:.2} , Specificity : {:.2} , Sensitivity : {:.2}'.format(Kfold_cnt, Accuracy, roc_auc, Specificity, Sensitivity))
            print("True : 1 but 0 :")
            print(error_list['1_to_0'])
            print("True : 0 but 1 :")
            print(error_list['0_to_1'])
            print("===================================================================================================")
        

            if WANDBRUN:
                wb_run.log({
                            "CNN Accuracy" : Accuracy,
                            "CNN AUC" : roc_auc,
                            "CNN Specificity" : Specificity,
                            "CNN Sensitivity" : Sensitivity
                            })

# ML ===============================================================
            if RUNML:
                # 強分類器
                # 提取特徵圖
                print("================================= Catboots Training ==============================================")
                ML_train_loader = DataLoader(train, shuffle = np.True_)
                ML_val_loader = DataLoader(val, shuffle = True)

                feature_train_data, feature_train_label = load_feature(ML_train_loader, model)
                feature_val_data, feature_val_label = load_feature(ML_val_loader, model)

                predict, predict_Probability = catboots_fit(feature_train_data, feature_train_label, feature_val_data, feature_val_label, CATBOOTS_INTER, ACTBOOTS_DETPH)

                ML_roc_auc, compute_img = MyEstimator.compute_auc(feature_val_label, predict_Probability, CLASSNANE, logPath, mode = 'ML_' + str(Kfold_cnt))
                ML_Accuracy, ML_Specificity, ML_Sensitivity, error_list, confusion_img = MyEstimator.confusion(feature_val_label, predict, val_keyLabel=None, classes = CLASSNANE, logPath = logPath, mode ='ML_' + str(Kfold_cnt))

                if ML_roc_auc != -1:
                    ML_roc_auc = max(ML_roc_auc.values())

                print("Kfold : {} , Accuracy : {:.2} => {:.2} , AUC : {:.2} => {:.2}".format(Kfold_cnt, Accuracy, ML_Accuracy, roc_auc, ML_roc_auc))
                print("Specificity : {:.2} => {:.2} , Sensitivity : {:.2} => {:.2}".format(Specificity, ML_Specificity, Sensitivity, ML_Sensitivity))
                print("===================================================================================================")

                if WANDBRUN:
                    wb_run.log({
                                "ML Accuracy" : ML_Accuracy,
                                "ML AUC" : ML_roc_auc,
                                "ML Specificity" : ML_Specificity,
                                "ML Sensitivity" : ML_Sensitivity
                                })

                
                ML_total_true += feature_val_label.tolist()
                ML_total_pred += predict.tolist()
                ML_total_pred_score += predict_Probability.tolist()
            


# Kflod end ================================================
        # Kfold CNN 結束交叉驗證
        Accuracy, Specificity, Sensitivity, error_list, confusion_img = MyEstimator.confusion(total_true, total_pred, total_keyLabel, classes = CLASSNANE, logPath = logPath, mode = 'Kfold_CNN')
        roc_auc, compute_img = MyEstimator.compute_auc(total_true, total_pred_score, CLASSNANE, logPath, mode = 'Kfold_CNN')
        
        print("True : 1 but 0 :")
        print(error_list['1_to_0'])
        print("True : 0 but 1 :")
        print(error_list['0_to_1'])


        if roc_auc != -1:
                roc_auc = float(max(roc_auc))
        if WANDBRUN:
            wb_run.log({
                        "KFold_CNN_ML Accuracy" : Accuracy,
                        "KFold_CNN_ML AUC" : roc_auc,
                        "KFold_CNN_ML Specificity" : Specificity.item(),
                        "KFold_CNN_ML Sensitivity" : Sensitivity.item(),
                        "KFold_CNN_ML compute": [wandb.Image(compute_img)],
                        "KFold_CNN_ML confusion": [wandb.Image(confusion_img)]
                        })

        if RUNML:
            # Kfold ML 結束交叉驗證
            ML_Accuracy, ML_Specificity, ML_Sensitivity, error_list, compute_img = MyEstimator.confusion(ML_total_true, ML_total_pred, val_keyLabel=None, classes = CLASSNANE, logPath = logPath, mode = 'Kfold_ML')
            ML_roc_auc, confusion_img = MyEstimator.compute_auc(ML_total_true, ML_total_pred_score, CLASSNANE, logPath, mode = 'Kfold_ML')

            if roc_auc != -1:
                ML_roc_auc = float(max(ML_roc_auc))

            # print("===================================================================================================")
            # print('Kfold:N= : {} ,Total = Accuracy : {:.3} , Specificity : {:.2} , Sensitivity : {:.2}'.format(KFOLD_N, Accuracy, Specificity, Sensitivity))
            # print("Total AUC: ", roc_auc)
            # print("===================================================================================================")
            print("============================-== KFlod Finish =====================================================")
            print("Kfold:N= {} , Accuracy : {:.2} => {:.2} , AUC : {:.2} => {:.2}".format(KFOLD_N, Accuracy, ML_Accuracy, roc_auc, ML_roc_auc))
            print("Specificity : {:.2} => {:.2} , Sensitivity : {:.2} => {:.2}".format(Specificity.item(), ML_Specificity.item(), Sensitivity.item(), ML_Sensitivity.item()))
            print("===================================================================================================")

            # if roc_auc != -1:
            #     roc_auc = float(max(roc_auc))

            if WANDBRUN:
                wb_run.log({
                            "KFold_CNN_ML Accuracy" : ML_Accuracy,
                            "KFold_CNN_ML AUC" : ML_roc_auc,
                            "KFold_CNN_ML Specificity" : ML_Specificity.item(),
                            "KFold_CNN_ML Sensitivity" : ML_Sensitivity.item(),
                            "KFold_CNN_ML compute": [wandb.Image(compute_img)],
                            "KFold_CNN_ML confusion": [wandb.Image(confusion_img)]
                        })

    if WANDBRUN:
        wb_run.finish()
