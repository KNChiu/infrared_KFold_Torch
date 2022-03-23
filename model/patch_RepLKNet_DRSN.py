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


class Shrinkage(nn.Module):
    def __init__(self, gap_size, channel):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear (channel, 1), # probably NN. Linear (channel, channel)
            nn.Sigmoid(),
        )
    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)

        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)

        x = x.unsqueeze(2).unsqueeze(2)
        #Soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class RepLKNet(nn.Module):
    def __init__(self, dim, depth, kernel_size):
        super(RepLKNet, self).__init__()
        self.dim = dim
        self.depth = depth
        self.kernel_size = kernel_size

        self.large_kernel = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=3),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim)
            )

        self.small_kernel = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim)
            )
        self.Shrinkage = Shrinkage((1,1), 256)

    def forward(self, x):
        rest_x = x
        x = self.large_kernel(x)
        ShrinkageOut = self.Shrinkage(x)
        x = torch.mul(ShrinkageOut, x)
        x = rest_x + x
        x = self.small_kernel(x)

        return x






# def RepLKNet(dim, depth, kernel_size):
#     return nn.Sequential(   
#         *[nn.Sequential( 
#             Residual(nn.Sequential(
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                 nn.Conv2d(dim, dim, kernel_size=kernel_size, padding='same'),
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                 nn.BatchNorm2d(dim),
#             )), 
#             Residual(nn.Sequential(
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                 nn.GELU(),
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                 nn.BatchNorm2d(dim)
#             )), 
#         )for i in range(depth)], 
#     )


class PatchRepLKNetDRSN(nn.Module):
    def __init__(self, dim = 768, depth = 3, kernel_size = 7, patch_size = 16, n_classes = 2):
        super(PatchRepLKNetDRSN, self).__init__()
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

        self.downC = nn.Conv2d(dim, 256, kernel_size=1)
        self.RepLKNet = RepLKNet(256, depth, kernel_size)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, n_classes)

        # self.norm = nn.LayerNorm(dim, eps=1e-6) # final norm layer
        # self.head = nn.Linear(dim, n_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        # x = self.cm_layer(x)
        x = self.downC(x)
        
        x = self.RepLKNet(x)

        gap = torch.mean(x,1)       # 可視化層
        featureOut = x.mean([-2, -1])

        x = self.gap(x)
        x = self.flat(x)

        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)

        return output, gap, featureOut

if __name__ == '__main__':
    import torch
    import netron
    import torch.optim as optim

    # GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)
    model = PatchRepLKNetDRSN(dim = 768, depth = 4, kernel_size = 7, patch_size = 16, n_classes = 2)
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