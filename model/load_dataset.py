import torch
from torch.utils.data import Dataset
import cv2
import concurrent
import os
import numpy as np
import pickle


class LoadDataset(Dataset):
    def __init__(self, path_fold:str,mutPath:str,cpu_count:int,transform=None):
        self.path_fold = path_fold
        self.mutPath = mutPath
        self.cpu_count = cpu_count
        self.transform = transform
        self.Data_all = self.load_mut(self.path_fold,self.mutPath,self.cpu_count)

    

class MyDataset(Dataset):
    def __init__(self, path_fold:str,mutPath:str,cpu_count:int,transform=None):
        print("開始讀檔")
        self.data = []
        self.label = []
        self.key = []
        self.path_fold = path_fold
        self.mutPath = mutPath
        self.cpu_count = cpu_count
        self.transform = transform
        self.Data_all = self.load_mut(self.path_fold,self.mutPath,self.cpu_count)
        self._dict2tensor(self.Data_all)
        
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
        for key in data:
            label = 0 if key.split("_")[-1] == "Ischemia" else 1
            data_zs = data[key]
            self.data.append(data_zs)
            self.label.append(label)
            self.key.append(key)
        
        self.data = torch.Tensor(self.data).permute(0,3,1,2)
        self.label = torch.LongTensor(self.label)
        self.key = np.array(self.key)
    
    def loadimg(self, job_path):
        inputImg = cv2.imread(job_path)
        inputImg = cv2.resize(inputImg, (640, 640), interpolation=cv2.INTER_AREA)

        return inputImg, job_path

    def load_mut(self, path_fold:str,mutPath:str,cpu_count:int) -> dict:
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
                        future = list(executor.map(self.loadimg, job_path))
            except:
                print("\r 多進程失敗 使用單進程讀圖")
                future = list(map(self.loadimg, job_path))

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
