import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

def total_variation(img: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Total Variation according to [1].

    Args:
        img: the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

    Return:
         a scalar with the computer loss.

    Examples:
        # >>> total_variation(torch.ones(3, 4, 4))
        tensor(0.)

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       total_variation_denoising.html>`__.

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

    if len(img.shape) < 3 or len(img.shape) > 4:
        raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    reduce_axes = (-3, -2, -1)
    res1 = pixel_dif1.abs().sum(dim=reduce_axes)
    res2 = pixel_dif2.abs().sum(dim=reduce_axes)

    return res1 + res2



class TotalVariation(nn.Module):
    r"""Compute the Total Variation according to [1].

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N,)` or scalar.

    Examples:
        >>> tv = TotalVariation()
        >>> output = tv(torch.ones((2, 3, 4, 4), requires_grad=True))
        >>> output.data
        tensor([0., 0.])
        >>> output.sum().backward()  # grad can be implicitly created only for scalar outputs

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """

    def forward(self, img) -> torch.Tensor:
        return torch.mean(total_variation(img))


import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[35:40].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[40:45].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[45:51].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[51:55].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[], style_layers=[0,1,2]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.
    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
     """

    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        # Load the VGG19 model trained on the ImageNet dataset.
        vgg19 = models.vgg19(pretrained=True).eval()
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:40])
        # Freeze model parameters.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        # sr = self.transform(sr, mode='bilinear', size=(224, 224), align_corners=False)
        # hr = self.transform(hr, mode='bilinear', size=(224, 224), align_corners=False)
        # Standardized operations
        sr = sr.sub(self.mean).div(self.std)
        hr = hr.sub(self.mean).div(self.std)

        # Find the feature map difference between the two images
        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss