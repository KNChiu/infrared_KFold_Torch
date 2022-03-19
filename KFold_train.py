#%%
from functools import total_ordering
import os
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

from model.patch_convmix_convnext import PatchConvmixConvnext
from model.focal_loss import FocalLoss
import json

import pandas as pd
import seaborn as sns

import wandb
import time

import catboost as cb


#%%   
def get_img_from_fig(fig, dpi=100):                       # plt 轉為 numpy
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def confusion(y_true, y_pred, calsses, logPath=None):
    Accuracy = accuracy_score(y_true, y_pred)
    Specificity = 0.0
    Sensitivity = 0.0

    if len(calsses) == 2:
        cf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        if (TN + FP) != 0:
            Specificity = TN / (TN + FP)
        if (TP + FN) != 0:
            Sensitivity = TP / (TP + FN)
    else:
        cf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    if logPath:
        df_cm = pd.DataFrame(cf_matrix, calsses, calsses)
        plt.figure(figsize = (9,6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        plt.xlabel('Predict', fontsize=10)        
        plt.ylabel('True', fontsize=10)
        plt.title('Accuracy : {:.2f} | Specificity : {:.2f} | Sensitivity : {:.2f}'.format(Accuracy, Specificity, Sensitivity), fontsize=10)

        if WANDBRUN:
            plot_img_np = get_img_from_fig(plt)    # plt 轉為 numpy]
            wb_run.log({"confusion": [wandb.Image(plot_img_np)]})

        plt.savefig(logPath + "//" + "confusion.jpg", bbox_inches='tight')
        plt.close('all')

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

            plot_img_np = get_img_from_fig(plt)    # plt 轉為 numpy
            if WANDBRUN:
                wb_run.log({"compute_auc": [wandb.Image(plot_img_np)]})
            plt.savefig(logPath + "//" + "compute_auc.jpg", bbox_inches='tight')
            plt.close()
    else:
        roc_auc = -1.00

    return roc_auc

def fit_model(model, train_loader, val_loader, classes):
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    # loss_func = FocalLoss(class_num=3, alpha = torch.tensor([0.36, 0.56, 0.72]).to(device), gamma = 4)
    loss_func = FocalLoss(class_num=len(classes), alpha = None, gamma = 4)

    for epoch in range(EPOCH):
        # val
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
        
        # val
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
    # test
    model.eval()
    correct = 0
    y_true = []
    y_pred = []
    y_pred_score = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            pred = model(x.to(device))
            pred, gap, test_feature = pred

            y_pred_score += pred.tolist()

            # 計算是否正確
            pred = torch.max(pred.data, 1)[1] 
            correct += (pred == y.to(device)).sum()

            y_true += y.tolist()
            y_pred += pred.tolist()
            
        
        Accuracy, Specificity, Sensitivity = confusion(y_true, y_pred, classes)
        roc_auc = compute_auc(y_true, y_pred_score, classes)

        return Accuracy, roc_auc, Specificity, Sensitivity, y_true, y_pred, y_pred_score, gap

def load_feature(dataloader, model):
    feature, label = [], []
    for idx, (x, y) in enumerate(dataloader):
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
    SAVEPTH = False
    SAVEIDX = True
    WANDBRUN = True
    SEED = 42
    
    CLASSNANE = ['Ischemia', 'Infect']
    # CLASSNANE = ['Ischemia', 'Acutephase', 'Recoveryperiod']
    # CLASSNANE = ['class_1', 'class_2', 'class_3']

    KFOLD_N = 2
    EPOCH = 1
    BATCHSIZE = 16
    LR = 0.0001

    CATBOOTS_INTER = 100
    ACTBOOTS_DETPH = 1

    DATAPATH = r'C:\Data\外科溫度\裁切\已分訓練集\cut_kfold'
    # DATAPATH = r'C:\Data\外科溫度\裁切\已分訓練集\cut_3_kfold'
    # DATAPATH = r'C:\Data\胸大肌\data\3classes\CC\train'

    if SEED:
        '''設定隨機種子碼'''
        os.environ["PL_GLOBAL_SEED"] = str(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # 建立 log
    logPath = r"logs//" + str(time.strftime("%m%d_%H%M", time.localtime()))
    if not os.path.isdir(logPath):
        os.mkdir(logPath)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([transforms.Resize((640, 640)),
                                    transforms.ToTensor()])

    if ISKFOLD:
        dataset = ImageFolder(DATAPATH, transform)          # 輸入數據集
        kf = KFold(n_splits = KFOLD_N, shuffle = True)
        Kfold_cnt = 0
        # acc_array = []
        totlal_acc = 0
        total_true = []
        total_pred = []
        total_pred_score = []

        ML_totlal_acc = 0
        ML_total_true = []
        ML_total_pred = []
        ML_total_pred_score = []

        # KFOLD
        for train_idx, val_idx in kf.split(dataset):
            Kfold_cnt += 1

            if WANDBRUN:
                wb_run = wandb.init(project='infraredThermal_kfold', entity='y9760210', reinit=True, group="KFold_1", name=str("kfold_N="+str(Kfold_cnt)))
            
            if SAVEIDX:
                json_dict = {'Kfold_cnt' : Kfold_cnt, 'val_idx': val_idx.tolist()}
                json_file = open(logPath + '//'+ 'kfold_idx.json', "a")
                json.dump(json_dict, json_file, indent=4)
                json_file.close()
            
            # 重組 kfold 數據集
            train = Subset(dataset, train_idx)
            val = Subset(dataset, val_idx)
        
            train_loader = DataLoader(train, batch_size = BATCHSIZE, shuffle = True, num_workers = 2)
            val_loader = DataLoader(val, batch_size = BATCHSIZE, shuffle = True, num_workers = 2)

            # 匯入模型
            model = PatchConvmixConvnext(dim = 768, depth = 3, kernel_size = 7, patch_size = 16, n_classes = len(CLASSNANE)).to(device)

            # Train
            fit_model(model, train_loader, val_loader, CLASSNANE)

            # Test
            Accuracy, roc_auc, Specificity, Sensitivity, kfold_true, kfold_pred, kfold_pred_score, gap = test_model(model, val_loader, CLASSNANE)

            if len(gap) > 8:
                cnt = 8
            else:
                cnt = len(gap)

            for i in range(cnt):
                plt.subplot(1, cnt, i+1)
                plt.imshow(gap[i].cpu().detach().numpy())   # 將注意力圖像取出
                plt.axis('off')         # 關閉邊框

            plot_img_np = get_img_from_fig(plt)    # plt 轉為 numpy
            plt.close('all')

            totlal_acc += Accuracy
            total_true += kfold_true
            total_pred += kfold_pred
            total_pred_score += kfold_pred_score

            if roc_auc != -1:
                roc_auc = max(roc_auc.values())

                
            # print("==================================== CNN Training=================================================")
            # print('Kfold : {} , Accuracy : {:.2e} , Test AUC : {:.2} , Specificity : {:.2} , Sensitivity : {:.2}'.format(Kfold_cnt, Accuracy, roc_auc, Specificity, Sensitivity))
            # print("===================================================================================================")
            
            if SAVEPTH:
                saveModelpath = logPath + "//" + str(Kfold_cnt) + "_last.pth"
                torch.save(model.state_dict(), saveModelpath)

            # 強分類器
            # 提取特徵圖
            print("================================= Catboots Training ==============================================")
            ML_train_loader = DataLoader(train, shuffle = np.True_)
            ML_val_loader = DataLoader(val, shuffle = True)
            feature_train_data, feature_train_label = load_feature(ML_train_loader, model)
            feature_val_data, feature_val_label = load_feature(ML_val_loader, model)
            predict, predict_Probability = catboots_fit(feature_train_data, feature_train_label, feature_val_data, feature_val_label, CATBOOTS_INTER, ACTBOOTS_DETPH)

            ML_roc_auc = compute_auc(feature_val_label, feature_val_data, CLASSNANE)
            ML_Accuracy, ML_Specificity, ML_Sensitivity = confusion(feature_val_label, predict, CLASSNANE, logPath)

            if ML_roc_auc != -1:
                ML_roc_auc = max(ML_roc_auc.values())

            # print("===================================================================================================")
            # print('Accuracy : {:.2e} , Test AUC : {:.2} , Specificity : {:.2} , Sensitivity : {:.2}'.format(ML_Accuracy, ML_roc_auc, ML_Specificity, ML_Sensitivity))
            # print("===================================================================================================")

            # print("===================================================================================================")
            print("Kfold : {} , Accuracy : {:.2} => {:.2} , AUC : {:.2} => {:.2}".format(Kfold_cnt, Accuracy, ML_Accuracy, roc_auc, ML_roc_auc))
            print("Specificity : {:.2} => {:.2} , Sensitivity : {:.2} => {:.2}".format(Specificity, ML_Specificity, Sensitivity, ML_Sensitivity))
            print("===================================================================================================")

            if WANDBRUN:
                wb_run.log({
                            "CNN Accuracy" : Accuracy,
                            "CNN AUC" : roc_auc,
                            "CNN Specificity" : Specificity,
                            "CNN Sensitivity" : Sensitivity,

                            "ML Accuracy" : ML_Accuracy,
                            "ML AUC" : ML_roc_auc,
                            "ML Specificity" : ML_Specificity,
                            "ML Sensitivity" : ML_Sensitivity
                        })
                wb_run.log({"image": [wandb.Image(plot_img_np)]})   # 將可視化上傳 wandb
            


            ML_totlal_acc += Accuracy.tolist()
            ML_total_true += feature_val_label.tolist()
            ML_total_pred += predict.tolist()
            ML_total_pred_score += predict_Probability.tolist()
            
        # Kfold CNN 結束交叉驗證
        Accuracy, Specificity, Sensitivity = confusion(total_true, total_pred, CLASSNANE, logPath)
        roc_auc = compute_auc(total_true, total_pred_score, CLASSNANE, logPath)

        # Kfold ML 結束交叉驗證
        ML_Accuracy, ML_Specificity, ML_Sensitivity = confusion(ML_total_true, ML_total_pred, CLASSNANE, logPath)
        ML_roc_auc = compute_auc(ML_total_true, ML_total_pred_score, CLASSNANE, logPath)

        if roc_auc != -1:
            roc_auc = float(max(roc_auc))
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
                        "KFold_CNN=>ML Accuracy" : Accuracy,
                        "KFold_CNN=>ML AUC" : roc_auc,
                        "KFold_CNN=>ML Specificity" : Specificity.item(),
                        "KFold_CNN=>ML Sensitivity" : Sensitivity.item()
                    })

            wb_run.log({
                        "KFold_CNN=>ML Accuracy" : ML_Accuracy,
                        "KFold_CNN=>ML AUC" : ML_roc_auc,
                        "KFold_CNN=>ML Specificity" : ML_Specificity.item(),
                        "KFold_CNN=>ML Sensitivity" : ML_Sensitivity.item()
                    })
            wb_run.finish()

    if WANDBRUN:
        wb_run.finish()

