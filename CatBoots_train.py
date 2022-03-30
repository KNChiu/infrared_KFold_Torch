#%%
import json
import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from model.patch_convmix_Attention import PatchConvMixerAttention
import json

import catboost as cb

from model.load_dataset import MyDataset
from model.assessment_tool import MyEstimator

#%%
def load_feature(dataloader, model, depth):
    feature, label, keyLabel = [], [], []
    for idx, (x, y, key) in enumerate(dataloader):
        # _, _, featureOut = model(x.to(device))
        keyLabel += key
        x = model.patch_embed(x.to(device))
        x = model.downC(x)
        for i in range(depth):
            x = model.cm_layer[i](x)
        featureOut = x.mean([-2, -1])

        featureOut = featureOut[0].to('cpu').detach().numpy()
        featureOut = featureOut.reshape((1, -1))[0]

        feature.append(featureOut)
        label.append(y.to('cpu').detach().numpy())
        
    feature = np.array(feature)
    label = np.array(label)
    return feature, label, keyLabel

def catboots_fit(train_data, train_label, val_data, val_label, iterations, CatBoost_depth=6):
    # cbc = cb.CatBoostClassifier(random_state=SEED, use_best_model=True, iterations=iterations, depth = CatBoost_depth,random_seed=SEED)
    # cbc = cb.CatBoostClassifier(iterations=10000,learning_rate=0.1,max_depth=7,verbose=100,
    #                                   early_stopping_rounds=500,task_type='GPU',eval_metric='AUC',random_seed=SEED)
    cbc = cb.CatBoostClassifier(
                               loss_function='MultiClass',
                                eval_metric='WKappa',
                               task_type="GPU",
                            #    learning_rate=0.01,
                               iterations=iterations,
                               od_type="Iter",
                                #depth=4,
                               early_stopping_rounds=500,
                                #l2_leaf_reg=10,
                                #border_count=96,
                               random_seed=42,
                                use_best_model=True
                              )

    cbc.fit(train_data, train_label,
            eval_set = [(val_data, val_label)],
            verbose=False,
            plot=False
            )
    predict = cbc.predict(val_data)
    predict_Probability = cbc.predict(val_data, prediction_type='Probability')
    return predict, predict_Probability

#%%
CLASSNANE = ['Ischemia', 'Infect']
SEED = 42
CNN_DETPH = 3
KERNELSIZE = 7
LOGPATH = r'C:\Data\surgical_temperature\trainingLogs\\'
DATAPATH = r'C:\Data\surgical_temperature\cut\classification\cut_96\\'
# DATAPATH = r'C:\Data\外科溫度\裁切\已分訓練集\cut_3_kfold'
# DATAPATH = r'C:\Data\胸大肌\data\3classes\CC\train'

MyEstimator = MyEstimator()
D = MyDataset(DATAPATH, LOGPATH, 2)
logPath = r'C:\Data\surgical_temperature\trainingLogs\logs\cos_150'
KFOLD_N = 10
# KFOLD_CNT = 3

CATBOOTS_INTER = 1000
# LOAD_FEATUR_DETPH = 2
# ACTBOOTS_DETPH = 2

if SEED:
    '''設定隨機種子碼'''
    os.environ["PL_GLOBAL_SEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

for loadDetph in range(CNN_DETPH):

    ML_total_true = []
    ML_total_pred = []
    ML_total_pred_score = []
    total_keyLabel = []

    for i in range(KFOLD_N):
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


        model = PatchConvMixerAttention(dim = 768, depth = CNN_DETPH, kernel_size = KERNELSIZE, patch_size = 16, n_classes = len(CLASSNANE)).to(device)

        model.eval()
        model.load_state_dict(torch.load(WEIGHTPATH))
        model.to(device)

        feature_train_data, feature_train_label, train_keyLabel = load_feature(ML_train_loader, model, depth=loadDetph+1)
        feature_val_data, feature_val_label, val_keyLabel = load_feature(ML_val_loader, model, depth=loadDetph+1)
        total_keyLabel += val_keyLabel

        predict, predict_Probability = catboots_fit(feature_train_data, feature_train_label, feature_val_data, feature_val_label, CATBOOTS_INTER)

        ML_roc_auc, _ = MyEstimator.compute_auc(feature_val_label, predict_Probability, CLASSNANE, logPath+"\\img", mode = 'ML_' + str(KFOLD_CNT))
        ML_Accuracy, ML_Specificity, ML_Sensitivity, error_list, _ = MyEstimator.confusion(feature_val_label, predict, val_keyLabel, CLASSNANE, logPath+"\\img", mode ='ML_' + str(KFOLD_CNT))

        ML_total_true += feature_val_label.tolist()
        ML_total_pred += predict.tolist()
        ML_total_pred_score += predict_Probability.tolist()

        if ML_roc_auc == -1:
            print("=================================================================================")
            print("KFOLD_CNT : {} , Accuracy : {:.2} ".format(KFOLD_CNT, ML_Accuracy))
            print("Specificity : {:.2} , Sensitivity : {:.2}".format(ML_Specificity, ML_Sensitivity))
        else:
            print("=================================================================================")
            print("KFOLD_CNT : {} , Accuracy : {:.2} , AUC : [{:.2}, {:.2}]".format(KFOLD_CNT, ML_Accuracy, ML_roc_auc[0], ML_roc_auc[1]))
            print("Specificity : {:.2} , Sensitivity : {:.2}".format(ML_Specificity, ML_Sensitivity))
        # print("=================================================================================")
        print("True : 1 but 0 :")
        print(error_list['1_to_0'])
        print("True : 0 but 1 :")
        print(error_list['0_to_1'])

    ML_Accuracy, ML_Specificity, ML_Sensitivity, error_list, _ = MyEstimator.confusion(ML_total_true, ML_total_pred, total_keyLabel, CLASSNANE, logPath+"\\img", mode = 'Kfold_ML_ALL')
    ML_roc_auc,_ = MyEstimator.compute_auc(ML_total_true, ML_total_pred_score, CLASSNANE, logPath+"\\img", mode = 'Kfold_ML_ALL')
    print("=================================================================================")
    print("*********************************************************************************")
    print("Load Detphs : {} , Accuracy : {:.2} , AUC : [{:.2}, {:.2}]".format(loadDetph+1, ML_Accuracy, ML_roc_auc[0], ML_roc_auc[1]))
    print("Specificity : {:.2} , Sensitivity : {:.2}".format(ML_Specificity, ML_Sensitivity))

    print("True : 1 but 0 :")
    print(error_list['1_to_0'])
    print("True : 0 but 1 :")
    print(error_list['0_to_1'])
    print("*********************************************************************************")


# %%
