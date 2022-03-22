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

from model.patch_convmix_convnext import PatchConvmixConvnext
from model.patch_RepLKNet_DRSN import PatchRepLKNetDRSN
from model.patch_RepLKNet_Attention import PatchRepLKNetAttention


from model.focal_loss import FocalLoss
import json

import pandas as pd
import seaborn as sns

import wandb
import time

import catboost as cb

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
# path = r'C:/Data/cut_kfold/'

# data_all = load_mut(path, 2)


#%%   
def get_img_from_fig(fig, dpi=100):                       # plt 轉為 numpy
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.close('all')
    return img

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
        plt.close("all")
        df_cm = pd.DataFrame(cf_matrix, calsses, calsses)
        plt.figure(figsize = (9,6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        plt.xlabel('Predict', fontsize=10)        
        plt.ylabel('True', fontsize=10)
        plt.title(str(mode) + '_Acc : {:.2f} | Specificity : {:.2f} | Sensitivity : {:.2f}'.format(Accuracy, Specificity, Sensitivity), fontsize=10)

        if WANDBRUN:
            plot_img_np = get_img_from_fig(plt)    # plt 轉為 numpy]
            wb_run.log({"confusion": [wandb.Image(plot_img_np)]})

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
            plt.close("all")
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

            plot_img_np = get_img_from_fig(plt)    # plt 轉為 numpy
            if WANDBRUN:
                wb_run.log({"compute_auc": [wandb.Image(plot_img_np)]})
            plt.savefig(logPath + "//" + str(mode) +"_compute_auc.jpg", bbox_inches='tight')
            plt.close("all")
    else:
        roc_auc = -1.00

    return roc_auc

def fit_model(model, train_loader, val_loader, classes):
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    # optimizer = torch.optim.SGD(model.parameters(), lr = LR)
    # loss_func = FocalLoss(class_num=3, alpha = torch.tensor([0.36, 0.56, 0.72]).to(device), gamma = 4)
    loss_func = FocalLoss(class_num=len(classes), alpha = None, gamma = 4)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)    # (1 + T_mult + T_mult**2) * T_0 // 5,15,35,75,155
    # print("1")
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
            
        train_roc_auc = compute_auc(train_y_true, train_y_pred_score, classes)
        train_Accuracy, Specificity, Sensitivity = confusion(train_y_true, train_y_pred, classes)
        
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

            roc_auc = compute_auc(y_true, y_pred_score, classes)
            Accuracy, Specificity, Sensitivity = confusion(y_true, y_pred, classes)

            
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
                                 # 將可視化上傳 wandb
                    })

            if epoch % 20 == 0:    
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
                plot_img_np = get_img_from_fig(plt)    # plt 轉為 numpy
                plt.close('all')
                
                wb_run.log({"val image": [wandb.Image(plot_img_np)]})   # 將可視化上傳 wandb

        print('  => Epoch : {}  Training Loss : {:.4e}  Val Loss : {:.4e}  Val ACC : {:.2}  Val AUC : {:.2}'.format(epoch + 1, training_loss, val_loss, Accuracy, roc_auc))

    return training_loss, val_loss 

def test_model(model, test_loader, classes):
    # test
    model.eval()
    # correct = 0
    y_true = []
    y_pred = []
    y_pred_score = []

    with torch.no_grad():
        for idx, (x, y, _) in enumerate(test_loader):
            pred = model(x.to(device))
            pred, gap, test_feature = pred

            y_pred_score += pred.tolist()

            # 計算是否正確
            pred = torch.max(pred.data, 1)[1] 
            # correct += (pred == y.to(device)).sum()

            y_true += y.tolist()
            y_pred += pred.tolist()
            
        
        Accuracy, Specificity, Sensitivity = confusion(y_true, y_pred, classes)
        roc_auc = compute_auc(y_true, y_pred_score, classes)

        return Accuracy, roc_auc, Specificity, Sensitivity, y_true, y_pred, y_pred_score, gap

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

def draw_gap(gap):
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
    return plot_img_np

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
    RUNML = False
    SEED = 42
    
    CLASSNANE = ['Ischemia', 'Infect']
    # CLASSNANE = ['Ischemia', 'Acutephase', 'Recoveryperiod']
    # CLASSNANE = ['class_1', 'class_2', 'class_3']
    CNN_DETPH = 1
    KFOLD_N = 82
    EPOCH = 300
    BATCHSIZE = 16
    LR = 0.001

    CATBOOTS_INTER = 200
    ACTBOOTS_DETPH = 1

    LOGPATH = r'C:\Data\surgical_temperature\trainingLogs\\'
    DATAPATH = r'C:\Data\surgical_temperature\cut\classification\cut_3_2_kfold\\'
    # DATAPATH = r'C:\Data\外科溫度\裁切\已分訓練集\cut_3_kfold'
    # DATAPATH = r'C:\Data\胸大肌\data\3classes\CC\train'

    D = MyDataset(load_mut(DATAPATH, LOGPATH, 2))


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
            model = PatchRepLKNetAttention(dim = 768, depth = CNN_DETPH, kernel_size = 15, patch_size = 16, n_classes = len(CLASSNANE)).to(device)

            # Train
            fit_model(model, train_loader, val_loader, CLASSNANE)

            # Test
            Accuracy, roc_auc, Specificity, Sensitivity, kfold_true, kfold_pred, kfold_pred_score, gap = test_model(model, val_loader, CLASSNANE)

            total_true += kfold_true
            total_pred += kfold_pred
            total_pred_score += kfold_pred_score

            if roc_auc != -1:
                roc_auc = max(roc_auc.values())

            print("==================================== CNN Training=================================================")
            print('Kfold : {} , Accuracy : {:.2e} , Test AUC : {:.2} , Specificity : {:.2} , Sensitivity : {:.2}'.format(Kfold_cnt, Accuracy, roc_auc, Specificity, Sensitivity))
            print("===================================================================================================")
            
            if SAVEPTH:
                saveModelpath = logPath + "//" + str(Kfold_cnt) + "_last.pth"
                torch.save(model.state_dict(), saveModelpath)

            if WANDBRUN:
                wb_run.log({
                            "CNN Accuracy" : Accuracy,
                            "CNN AUC" : roc_auc,
                            "CNN Specificity" : Specificity,
                            "CNN Sensitivity" : Sensitivity,
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

                ML_roc_auc = compute_auc(feature_val_label, predict_Probability, CLASSNANE, logPath, mode = 'ML_' + str(Kfold_cnt))
                ML_Accuracy, ML_Specificity, ML_Sensitivity = confusion(feature_val_label, predict, CLASSNANE, logPath, mode ='ML_' + str(Kfold_cnt))

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
        Accuracy, Specificity, Sensitivity = confusion(total_true, total_pred, CLASSNANE, logPath, mode = 'Kfold_CNN')
        roc_auc = compute_auc(total_true, total_pred_score, CLASSNANE, logPath, mode = 'Kfold_CNN')
        
        if roc_auc != -1:
                roc_auc = float(max(roc_auc))
        if WANDBRUN:
            wb_run.log({
                        "KFold_CNN=>ML Accuracy" : Accuracy,
                        "KFold_CNN=>ML AUC" : roc_auc,
                        "KFold_CNN=>ML Specificity" : Specificity.item(),
                        "KFold_CNN=>ML Sensitivity" : Sensitivity.item()
                    })


        if RUNML:
            # Kfold ML 結束交叉驗證
            ML_Accuracy, ML_Specificity, ML_Sensitivity = confusion(ML_total_true, ML_total_pred, CLASSNANE, logPath, mode = 'Kfold_ML')
            ML_roc_auc = compute_auc(ML_total_true, ML_total_pred_score, CLASSNANE, logPath, mode = 'Kfold_ML')

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
                            "KFold_CNN=>ML Accuracy" : ML_Accuracy,
                            "KFold_CNN=>ML AUC" : ML_roc_auc,
                            "KFold_CNN=>ML Specificity" : ML_Specificity.item(),
                            "KFold_CNN=>ML Sensitivity" : ML_Sensitivity.item()
                        })

    if WANDBRUN:
        wb_run.finish()

#%%

# import concurrent
# import tqdm
# from torch.utils.data import Dataset


# def loadimg(job_path):
#     inputImg = cv2.imread(job_path)
#     inputImg = cv2.resize(inputImg, (640, 640), interpolation=cv2.INTER_AREA)

#     return inputImg, job_path

# def load_mut(path_fold:str,cpu_count:int) -> dict:
#     import pickle
    
#         # path_fold => r'C:\Users\sk\Desktop\sanzuo\transform\\'
    
#     if os.path.isfile('load_mut.json'):
#         print("使用暫存檔讀資料")
#         with open('load_mut.json', 'rb') as fp:
#             Data_all = pickle.load(fp)
#         return Data_all
#     else:
#         print("站存檔不存在，重新創建")
#         job_path = []
#         Data_all = {}

#         for classes in ['0_Ischemia','1_Infect']:
#             f = path_fold+classes+'/'
#             job_path += [f+img_name for img_name in os.listdir(f)]
#             # print(job_path)
#         try:
#             with concurrent.futures.ProcessPoolExecutor(cpu_count) as executor: ## 默认为1
#                     future = list(executor.map(loadimg, job_path))
#         except:
#             print("\r 多進程失敗 使用單進程讀圖")
#             future = list(map(loadimg, job_path))

#         for img, file_path in future:
#             path, img_name = os.path.split(file_path)
#             _, classes = os.path.split(path)
#             key = img_name.split('_')[0] + '_' + classes

#             if key in Data_all:
#                 Data_all[key].append(img)
#             else:
#                 Data_all[key] = img   

#         # 寫入暫存檔
#         with open('load_mut.json', 'wb') as fp:
#             pickle.dump(Data_all, fp)
        
#         return Data_all


# class MyDataset(Dataset):
#     def __init__(self,data:dict):
#         self.data = []
#         self.label = []
#         self.key = []
#         self._dict2tensor(data)

#     def __getitem__(self,index):
#         """_summary_

#         Args:
#             index (_type_): description

#         Returns:
#             list[list,list,list]: [data,label,key]
#         """
#         return self.data[index],self.label[index],self.key[index]

#     def __len__(self):
#         return len(self.data)
    
#     def _dict2tensor(self,data:dict) -> None:
#         # 1=>NG 0=>G
#         to_numpy = np.array
#         from sklearn import preprocessing
#         zscore = preprocessing.StandardScaler
#         for key in data:
#             # print(key)
#             label = 0 if key.split("_")[-1] == "0_Ischemia" else 1
#             # print(data[key])
#             # data_numpy = to_numpy(data[key])
#             # data_zs = zscore().fit_transform(data_numpy)
#             data_zs = data[key]
#             self.data.append(data_zs)
#             self.label.append(label)
#             self.key.append(key)
        
#         self.data = torch.Tensor(self.data).permute(0,3,1,2)
#         self.label = torch.LongTensor(self.label)
#         self.key = np.array(self.key)



# # path = r'C:/Data/cut_kfold/'

# # data_all = load_mut(path, 2)
# path = r'C:/Data/cut_kfold/'
# D = MyDataset(load_mut(path, 2))


# # %%
