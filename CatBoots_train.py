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

from model.patch_convmix_convnext import PatchConvmixConvnext
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
    for idx, (x, y) in enumerate(dataloader):
        # _, _, featureOut = model(x.to(device))

        x = model.patch_embed(x.to(device))
        for i in range(depth):
            x = model.cm_layer[i](x)
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
            plt.close()
    else:
        roc_auc = -1.00

    return roc_auc


#%%
CLASSNANE = ['Ischemia', 'Infect']
DATAPATH = r'C:\Data\外科溫度\裁切\已分訓練集\cut_kfold'
SEED = 42

# KFOLD_CNT = 3

CATBOOTS_INTER = 200
ACTBOOTS_DETPH = 1
FEATUR_DETPH = 1


if SEED:
    '''設定隨機種子碼'''
    os.environ["PL_GLOBAL_SEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

logPath = r'logs\0320_1203'

for i in range(10):
    KFOLD_CNT = i+1
    WEIGHTPATH = r"logs\\0320_1203\\" + str(KFOLD_CNT) + '_last.pth'
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

    dataset = ImageFolder(DATAPATH, transform)          # 輸入數據集

    train = Subset(dataset, train_idx)
    val = Subset(dataset, val_idx)

    ML_train_loader = DataLoader(train, shuffle = np.True_)
    ML_val_loader = DataLoader(val, shuffle = True)


    model = PatchConvmixConvnext(dim = 768, depth = 3, kernel_size = 7, patch_size = 16, n_classes = len(CLASSNANE))

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

    print("=================================================================================")
    print("KFOLD_CNT : {} , Accuracy : {:.2} , AUC : [{:.2}, {:.2}]".format(KFOLD_CNT, ML_Accuracy, ML_roc_auc[0], ML_roc_auc[1]))
    print("Specificity : {:.2} , Sensitivity : {:.2}".format(ML_Specificity, ML_Sensitivity))
    print("=================================================================================")


# %%
