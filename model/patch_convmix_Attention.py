#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def convmixer_layer(dim, depth, kernel_size, train_mode=0):
    if train_mode == 0:
        return nn.Sequential(   
            *[nn.Sequential( 
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, padding='same', dilation=2, groups=dim),
                    ChannelAttention(dim),
                    nn.Conv2d(dim, dim, kernel_size=3, padding='same', dilation=2, groups=dim),
                    ChannelAttention(dim),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )), 
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1),
                    ChannelAttention(dim),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )), 
            )for dilation_rate in range(depth)], 
        )
    elif train_mode == 1:
        return nn.Sequential(   
            *[nn.Sequential( 
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, padding='same', dilation=2, groups=dim),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size=3, padding='same', dilation=2, groups=dim),
                    nn.GELU(),
                    ChannelAttention(dim),
                    nn.BatchNorm2d(dim),
                )), 
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    ChannelAttention(dim),
                    nn.BatchNorm2d(dim)
                )), 
            )for dilation_rate in range(depth)], 
        )
    elif train_mode == 2:
        return nn.Sequential(   
            *[nn.Sequential( 
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, padding='same', dilation=2, groups=dim),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size=1, padding='same', dilation=2, groups=dim),
                    nn.GELU(),
                    ChannelAttention(dim),
                    nn.BatchNorm2d(dim),
                )), 
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    ChannelAttention(dim),
                    nn.BatchNorm2d(dim)
                )), 
            )for dilation_rate in range(depth)], 
        )

class PatchConvMixerAttention(nn.Module):
    def __init__(self, dim = 768, depth = 3, kernel_size = 7, patch_size = 16, n_classes = 2, train_mode = 0):
        super(PatchConvMixerAttention, self).__init__()
        self.dim = dim
        self.depth = depth
        self.n_class = n_classes
        self.kernel_size = kernel_size

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size = patch_size, stride = patch_size),
            nn.GELU(),
            # nn.LeakyReLU(),
            nn.BatchNorm2d(dim)
        )
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()

        self.downC = nn.Conv2d(dim, dim, kernel_size=1)
        self.cm_layer = convmixer_layer(dim, depth, kernel_size, train_mode)


        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.flat = nn.Flatten()

        self.fc = nn.Linear(dim, n_classes)

    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.downC(x)
        x = self.cm_layer(x)

        x = self.ca(x) * x
        # x = self.sa(x) * x

        gap = torch.mean(x,1)       # 可視化層
        featureOut = x.mean([-2, -1])   

        x = self.gap(x)
        x = self.flat(x)

        output = self.fc(x)

        return output, gap, featureOut

if __name__ == '__main__':
    import torch
    import netron
    import torch.optim as optim

    # GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)
    model = PatchConvMixerAttention(dim = 768, depth = 3, kernel_size = 7, patch_size = 16, n_classes = 2)
    # print(model)

    heat_visual = False

    if heat_visual:
        from cgi import test
        import cv2
        import torch
        from torchvision import transforms
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder

        import numpy as np
        import matplotlib.pyplot as plt

        WEIGHTPATH = r'G:\我的雲端硬碟\Lab\Project\外科溫度\infraredthermal_PL\log\best_check_point\best.pth'
        DATAPATH = r"G:\我的雲端硬碟\Lab\Project\外科溫度\infraredthermal_PL\log\輸入測試\\"
        # SAVEPATH = r'G:\我的雲端硬碟\Lab\Project\外科溫度\infraredthermal_PL\log\patchimg'
        SAVEPATH = None

        transform = transforms.Compose([transforms.Resize((640, 640)),
                                    transforms.ToTensor(),])


        model.eval()
        model.load_state_dict(torch.load(WEIGHTPATH))
        model.to(device)


        test = ImageFolder(DATAPATH + 'test', transform)
        test_loader = DataLoader(test, shuffle = False)
        
        for idx, (x, y) in enumerate(test_loader):
            # cam_visualize(idx, model, x.to(device))
            x = model.patch_embed(x.to(device))
            x = x[0].permute(0, 1, 2).to('cpu').detach().numpy()
            for i in range(x.shape[0]):
                # plt.imshow(x[i])
                if SAVEPATH:
                    plt.imsave(SAVEPATH + '\\' + str(i) + '.jpg', x[i])
                plt.show()

    else:
        _input = torch.randn(2, 3, 640, 640)
        s = model(_input)

        # print(s[1][0].detach().numpy().shape)
        # plt.imshow(s[1][0].detach().numpy())        # 顯示特徵萃取效果
        # plt.show()

        # _input = _input.unsqueeze(0)
        # # print(_input.size())
        onxx_path = r'model.onnx'
        torch.onnx.export(model, _input, onxx_path)

        netron.start(onxx_path)