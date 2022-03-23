#%%
import json
from collections import defaultdict

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

from model.patch_RepLKNet_Attention import PatchRepLKNetAttention
from model.focal_loss import FocalLoss
import json

import pandas as pd
import seaborn as sns

import wandb
import time

import catboost as cb

#%%
def load_feature(dataloader, model, depth):
    feature, label = [], []
    for idx, (x, y, _) in enumerate(dataloader):
        # _, _, featureOut = model(x.to(device))

        x = model.patch_embed(x.to(device))
        x = model.downC(x)
        for i in range(depth):
            x = model.RepLKNet[i](x)
        featureOut = x.mean([-2, -1])

        featureOut = featureOut[0].to('cpu').detach().numpy()
        featureOut = featureOut.reshape((1, -1))[0]

        feature.append(featureOut)
        label.append(y.to('cpu').detach().numpy())
        
    feature = np.array(feature)
    label = np.array(label)
    return feature, label

def catboots_fit(train_data, train_label, val_data, val_label, iterations, CatBoost_depth):
    cbc = cb.CatBoostClassifier(random_state=SEED, use_best_model=True, iterations=iterations, depth = CatBoost_depth)
    cbc.fit(train_data, train_label,
            eval_set = [(val_data, val_label)],
            verbose=False,
            plot=False
            )
    predict = cbc.predict(val_data)
    predict_Probability = cbc.predict(val_data, prediction_type='Probability')
    return predict, predict_Probability

def confusion(y_true, y_pred, calsses, logPath=None, mode = ''):
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
        plt.title(str(mode) + '_Acc : {:.2f} | Specificity : {:.2f} | Sensitivity : {:.2f}'.format(Accuracy, Specificity, Sensitivity), fontsize=10)

        # if WANDBRUN:
        #     plot_img_np = get_img_from_fig(plt)    # plt 轉為 numpy]
        #     wb_run.log({"confusion": [wandb.Image(plot_img_np)]})

        plt.savefig(logPath + "//" + str(mode) + "_confusion.jpg", bbox_inches='tight')
        plt.close('all')

    return Accuracy, Specificity, Sensitivity

def compute_auc(y_true, y_score, classes, logPath=None, mode = ''):
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
            plt.title(str(mode) + '_multi-calss ROC')
            plt.legend(loc="lower right")

            # plot_img_np = get_img_from_fig(plt)    # plt 轉為 numpy
            # if WANDBRUN:
            #     wb_run.log({"compute_auc": [wandb.Image(plot_img_np)]})
            plt.savefig(logPath + "//" + str(mode) +"_compute_auc.jpg", bbox_inches='tight')
            plt.close('all')
    else:
        roc_auc = -1.00

    return roc_auc

#%%
import concurrent
from torch.utils.data import Dataset

def loadimg(job_path):
    inputImg = cv2.imread(job_path)
    inputImg = cv2.resize(inputImg, (640, 640), interpolation=cv2.INTER_AREA)

    return inputImg, job_path

def load_mut(path_fold:str,mutPath:str,cpu_count:int) -> dict:
    import pickle
    
        # path_fold => r'C:\Users\sk\Desktop\sanzuo\transform\\'
    
    if os.path.isfile(mutPath + '\load_mut.json'):
        print("使用暫存檔讀資料")
        with open(mutPath + '\load_mut.json', 'rb') as fp:
            Data_all = pickle.load(fp)
        return Data_all
    else:
        print("站存檔不存在，重新創建")
        job_path = []
        Data_all = {}

        for classes in ['0_Ischemia','1_Infect']:
            f = path_fold+classes+'/'

            for img_name in os.listdir(f):
                if (img_name.split('.')[-1]) == "jpg":
                    job_path += [f+img_name]
            # job_path += [f+img_name for img_name in os.listdir(f)]
        try:
            with concurrent.futures.ProcessPoolExecutor(cpu_count) as executor: ## 默认为1
                    future = list(executor.map(loadimg, job_path))
        except:
            print("\r 多進程失敗 使用單進程讀圖")
            future = list(map(loadimg, job_path))

        for img, file_path in future:
            path, img_name = os.path.split(file_path)
            _, classes = os.path.split(path)
            key = img_name.split('_')[0] + '_' + classes

            if key in Data_all:
                Data_all[key].append(img)
            else:
                Data_all[key] = img   

        # 寫入暫存檔
        with open(mutPath + '\load_mut.json', 'wb') as fp:
            pickle.dump(Data_all, fp)
        
        return Data_all

class MyDataset(Dataset):
    def __init__(self,data:dict):
        print("開始讀檔")
        self.data = []
        self.label = []
        self.key = []
        self._dict2tensor(data)
        print("讀檔完畢")


    def __getitem__(self,index):
        """_summary_

        Args:
            index (_type_): description

        Returns:
            list[list,list,list]: [data,label,key]
        """
        return self.data[index],self.label[index],self.key[index]

    def __len__(self):
        return len(self.data)
    
    def _dict2tensor(self,data:dict) -> None:
        # to_numpy = np.array
        # from sklearn import preprocessing
        # zscore = preprocessing.StandardScaler
        for key in data:
            # print(key)
            label = 0 if key.split("_")[-1] == "Ischemia" else 1
            # print(data[key])
            # data_numpy = to_numpy(data[key])
            # data_zs = zscore().fit_transform(data_numpy)
            data_zs = data[key]
            self.data.append(data_zs)
            self.label.append(label)
            self.key.append(key)
        
        self.data = torch.Tensor(self.data).permute(0,3,1,2)
        self.label = torch.LongTensor(self.label)
        self.key = np.array(self.key)
#%%
CLASSNANE = ['Ischemia', 'Infect']
SEED = 42
CNN_DETPH = 1
KERNELSIZE = 7
LOGPATH = r'C:\Data\surgical_temperature\trainingLogs\\'
DATAPATH = r'C:\Data\surgical_temperature\cut\classification\cut_3_2_kfold\\'
# DATAPATH = r'C:\Data\外科溫度\裁切\已分訓練集\cut_3_kfold'
# DATAPATH = r'C:\Data\胸大肌\data\3classes\CC\train'

D = MyDataset(load_mut(DATAPATH, LOGPATH, 2))

# KFOLD_CNT = 3

CATBOOTS_INTER = 1000
ACTBOOTS_DETPH = 4
FEATUR_DETPH = 1
if SEED:
    '''設定隨機種子碼'''
    os.environ["PL_GLOBAL_SEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

logPath = r'C:\Data\surgical_temperature\trainingLogs\logs\0322_2148'


ML_total_true = []
ML_total_pred = []
ML_total_pred_score = []


for i in range(10):
    KFOLD_CNT = i+1
    WEIGHTPATH = logPath + "\\" +str(KFOLD_CNT) + '_last.pth'
    with open(logPath + '\\kfold_idx.json', 'r') as f:
        data = json.load(f)

    # for i in range(10):
    #     print(data['Kfold_cnt' + str(i+1)])
    #     # print(data['Kfold_cnt' + str(i+1)]['train_idx'])

    train_idx = data['Kfold_cnt' + str(KFOLD_CNT)]['train_idx']
    val_idx = data['Kfold_cnt' + str(KFOLD_CNT)]['val_idx']


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([transforms.Resize((640, 640)),
                                    transforms.ToTensor()])

    # dataset = ImageFolder(DATAPATH, transform)          # 輸入數據集
    dataset = D 

    train = Subset(dataset, train_idx)
    val = Subset(dataset, val_idx)

    ML_train_loader = DataLoader(train, shuffle = np.True_)
    ML_val_loader = DataLoader(val, shuffle = True)


    model = PatchRepLKNetAttention(dim = 768, depth = CNN_DETPH, kernel_size = KERNELSIZE, patch_size = 16, n_classes = len(CLASSNANE)).to(device)

    model.eval()
    model.load_state_dict(torch.load(WEIGHTPATH))
    model.to(device)

    # x = model.patch_embed(_input)
    # for i in range(3):
    #     x = model.cm_layer[i](x)


    feature_train_data, feature_train_label = load_feature(ML_train_loader, model, depth=FEATUR_DETPH)
    feature_val_data, feature_val_label = load_feature(ML_val_loader, model, depth=FEATUR_DETPH)

    predict, predict_Probability = catboots_fit(feature_train_data, feature_train_label, feature_val_data, feature_val_label, CATBOOTS_INTER, ACTBOOTS_DETPH)

    ML_roc_auc = compute_auc(feature_val_label, predict_Probability, CLASSNANE, logPath+"\\img", mode = 'ML_' + str(KFOLD_CNT))
    ML_Accuracy, ML_Specificity, ML_Sensitivity = confusion(feature_val_label, predict, CLASSNANE, logPath+"\\img", mode ='ML_' + str(KFOLD_CNT))


    ML_total_true += feature_val_label.tolist()
    ML_total_pred += predict.tolist()
    ML_total_pred_score += predict_Probability.tolist()


    print("=================================================================================")
    print("KFOLD_CNT : {} , Accuracy : {:.2} , AUC : [{:.2}, {:.2}]".format(KFOLD_CNT, ML_Accuracy, ML_roc_auc[0], ML_roc_auc[1]))
    print("Specificity : {:.2} , Sensitivity : {:.2}".format(ML_Specificity, ML_Sensitivity))
    # print("=================================================================================")


ML_Accuracy, ML_Specificity, ML_Sensitivity = confusion(ML_total_true, ML_total_pred, CLASSNANE, logPath+"\\img", mode = 'Kfold_ML_ALL')
ML_roc_auc = compute_auc(ML_total_true, ML_total_pred_score, CLASSNANE, logPath+"\\img", mode = 'Kfold_ML_ALL')

print("---------------------------------------------------------------------------------")
print("KFOLD_N= : {} , Accuracy : {:.2} , AUC : [{:.2}, {:.2}]".format(KFOLD_CNT, ML_Accuracy, ML_roc_auc[0], ML_roc_auc[1]))
print("Specificity : {:.2} , Sensitivity : {:.2}".format(ML_Specificity, ML_Sensitivity))
print("=================================================================================")

# %%
