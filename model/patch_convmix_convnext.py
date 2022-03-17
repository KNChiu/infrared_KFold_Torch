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

def convmixer_layer(dim, depth, kernel_size):
    return nn.Sequential(   # 7x7+1x1
        *[nn.Sequential(
                Residual(nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding = 3),
                            nn.GELU(),
                            nn.BatchNorm2d(dim),
                        )),

                    nn.Conv2d(dim, dim, kernel_size=3, padding = 1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )),
                
        # )
        ) for i in range(depth)],
    )
    ''' 
    return nn.Sequential(   # 7x7+3x3+1x1
        *[Residual(nn.Sequential(
                Residual(nn.Sequential(
                    Residual(nn.Sequential(
                                nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = kernel_size, groups = dim, padding = 3),
                                nn.GELU(),
                            )),
                    nn.Conv2d(dim, dim, kernel_size=3, padding = 1),
                    nn.GELU(),
                    )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                
        )) for i in range(depth)],
    )
    '''

# class LayerNorm(nn.Module):
#     r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
#     with shape (batch_size, channels, height, width).
#     """
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError 
#         self.normalized_shape = (normalized_shape, )
    
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x

# class Block(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
    
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
#         self.norm = LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
#                                     requires_grad=True) if layer_scale_init_value > 0 else None
#         # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.drop_path = nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

#         x = input + self.drop_path(x)
#         return x



class PatchConvmixConvnext(nn.Module):
    def __init__(self, dim = 768, depth = 3, kernel_size = 7, patch_size = 16, n_classes = 2):
        super(PatchConvmixConvnext, self).__init__()
        self.dim = dim
        self.depth = depth
        self.n_class = n_classes
        # self.inside_dim = inside_dim
        self.kernel_size = kernel_size

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size = patch_size, stride = patch_size),
            nn.GELU(),
            # nn.LeakyReLU(),
            nn.BatchNorm2d(dim)
        )

        self.cm_layer = convmixer_layer(self.dim, self.depth, self.kernel_size)

        # self.convNeXt = nn.Sequential(
        #     *[Block(dim=dim) for i in range(3)]
        # )

        # self.gap = nn.AdaptiveAvgPool2d((1,1))
        # self.flat = nn.Flatten()

        self.fc1 = nn.Linear(dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, n_classes)

        # self.norm = nn.LayerNorm(dim, eps=1e-6) # final norm layer
        # self.head = nn.Linear(dim, n_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cm_layer(x)

        # x = self.convNeXt(x)

        gap = torch.mean(x,1)       # 可視化層

        # x = self.gap(x)
        # x = self.norm(x.mean([-2, -1]))             # global average pooling, (N, C, H, W) -> (N, C)
        x = x.mean([-2, -1])
        # x = self.flat(x)

        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        # output = self.head(x)
        return output, gap

if __name__ == '__main__':
    import torch
    import netron
    import torch.optim as optim

    # GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)
    model = PatchConvmixConvnext(dim = 768, depth = 4, kernel_size = 7, patch_size = 16, n_classes = 2)
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
        _input = torch.randn(1, 3, 640, 640)
        s = model(_input)

        # print(s[1][0].detach().numpy().shape)
        # plt.imshow(s[1][0].detach().numpy())        # 顯示特徵萃取效果
        # plt.show()

        # _input = _input.unsqueeze(0)
        # # print(_input.size())
        onxx_path = r'model.onnx'
        torch.onnx.export(model, _input, onxx_path)

        netron.start(onxx_path)