import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

normalizator = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
def normal2uint(im):
    return ((im+1.)/2. * 255.).type(torch.float32)
def uint2normal(im):
    return normalizator(im/255.)

def apply_pyramid(x, k=1.):
    # print(x.shape[-2:])
    # print((x.shape[-2]//(2**k), x.shape[2]//(2**k)))
    down =  T.Resize((int(x.shape[-2]//(2**k)), int(x.shape[-1]//(2**k))), interpolation=InterpolationMode.NEAREST)
    up = T.Resize(x.shape[-2:], interpolation=InterpolationMode.BICUBIC)
    return up(down(x))

# def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
#     kernel = torch.tensor([[1., 4., 6., 4., 1],
#                            [4., 16., 24., 16., 4.],
#                            [6., 24., 36., 24., 6.],
#                            [4., 16., 24., 16., 4.],
#                            [1., 4., 6., 4., 1.]])
#     kernel /= 256.
#     kernel = kernel.repeat(channels, 1, 1, 1)
#     kernel = kernel.to(device)
#     return kernel
# def pyrDown(x, k=1):
#     return T.Resize(x.shape[0]/(2**k), x.shape[1]/(2**k), interpolation=InterpolationMode.NEAREST)(x)
#
# def pyrUp(x, k=1):
#     return T.Resize(x.shape[0]*(2**k), x.shape[1]*(2**k), interpolation=InterpolationMode.NEAREST)(x)


# def pyrDown(x, k=1):
#     return x[:, :, ::2, ::2]
#
# def pyrUp(x,k=1):
#     cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
#     cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
#     cc = cc.permute(0,1,3,2)
#     cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
#     cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
#     x_up = cc.permute(0,1,3,2)
#     return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))
#
# def conv_gauss(img, kernel):
#     img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
#     out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
#     return out