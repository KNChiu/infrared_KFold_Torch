import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pad
# import cv2
from PIL import Image
import concurrent
import os
import numpy as np
import pickle

from queue import Queue
from threading import Thread

from torchvision import transforms
import numbers

def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class NewPad(object):
    def __init__(self, fill=0, padding_mode='edge'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        mean = int(np.mean(img))
        return pad(img, get_padding(img), mean, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)



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
        
        # self.data = torch.Tensor(np.array(self.data)).permute(0,3,1,2)
        # self.data = self.data
        self.label = torch.LongTensor(np.array(self.label))
        self.key = np.array(self.key)
    


    def loadimg(self, job_path):
        inputImg = Image.open(job_path)
        # outputImg = cv2.resize(inputImg, (640, 640), interpolation=cv2.INTER_AREA)
        transform1 = transforms.Compose([
                                    # NewPad(),
                                    transforms.Resize([640, 640]),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(
                                    #             mean=[0.485, 0.456, 0.406],
                                    #             std=[0.229, 0.224, 0.225])
                                    ])

        outputImg = transform1(inputImg)


        return outputImg, job_path

    def load_mut(self, path_fold:str,mutPath:str,cpu_count:int) -> dict:
        # if os.path.isfile(mutPath + '\load_mut.json'):
        #     print("使用暫存檔資料")
        #     try:
        #         with concurrent.futures.ProcessPoolExecutor(cpu_count) as executor: ## 默认为1
        #             with open(mutPath + '\load_mut.json', 'rb') as fp:
        #                 Data_all = pickle.load(fp)
        #             return Data_all
        #     except:
        #         print("\r 多進程失敗使用單進程讀暫存檔資料")
        #         with open(mutPath + '\load_mut.json', 'rb') as fp:
        #             Data_all = pickle.load(fp)
        #         return Data_all

        #     # with open(mutPath + '\load_mut.json', 'rb') as fp:
        #     #     Data_all = pickle.load(fp)
        #     # return Data_all
        # else:
        # print("站存檔不存在，重新創建")
        print("重新創建數據集")
        job_path = []
        Data_all = {}

        for classes in ['0_Infect', '1_Ischemia',]:
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


class CudaDataLoader:
    """ 异步预先将数据从CPU加载到GPU中 """

    def __init__(self, loader, device, queue_size=2):
        self.device = device
        self.queue_size = queue_size
        self.loader = loader

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()

    def load_loop(self):
        """ 不断的将cuda数据加载到队列里 """
        # The loop that will load into the queue in the background
        torch.cuda.set_device(self.device)
        while True:
            for i, sample in enumerate(self.loader):
                self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        """ 将batch数据从CPU加载到GPU中 """
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)
        elif sample is None or type(sample) == str:
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        # 加载线程挂了
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # 一个epoch加载完了
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        # 下一个batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

class _RepeatSampler(object):
    """ 一直repeat的sampler """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """ 多epoch训练时，DataLoader对象不用重新建立线程和batch_sampler对象，以节约每个epoch的初始化时间 """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)