import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


##########################################################################
## Losses ported from Phaseformer

class GradientLoss(nn.Module):
    """Laplacian gradient loss (same kernel as Phaseformer Gradient_Loss)."""

    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel_g = [[[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                    [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                    [[0, 1, 0], [1, -4, 1], [0, 1, 0]]]
        kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
        self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        gradient_pred   = F.conv2d(pred,   self.weight_g, groups=3)
        gradient_target = F.conv2d(target, self.weight_g, groups=3)
        return self.l1(gradient_pred, gradient_target)


class VGGPerceptualLoss(nn.Module):
    """VGG16-based perceptual loss (same as Phaseformer VGGPerceptualLoss)."""

    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        import torchvision
        try:
            # torchvision >= 0.13
            weights = torchvision.models.VGG16_Weights.DEFAULT
            vgg = torchvision.models.vgg16(weights=weights)
        except AttributeError:
            # torchvision < 0.13 fallback
            vgg = torchvision.models.vgg16(pretrained=True)
        blocks = [
            vgg.features[:4].eval(),
            vgg.features[4:9].eval(),
            vgg.features[9:16].eval(),
            vgg.features[16:23].eval(),
        ]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target, feature_layers=(0, 1, 2, 3)):
        pred   = (pred   - self.mean) / self.std
        target = (target - self.mean) / self.std
        pred   = F.interpolate(pred,   mode='bilinear', size=(224, 224), align_corners=False)
        target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x, y = pred, target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
        return loss


class MultiscaleLoss(nn.Module):
    """Softmax-weighted combination of Charbonnier, VGG Perceptual, Gradient,
    and MS-SSIM losses — mirrors Phaseformer's WeightedLoss(4) approach.

    The 4 scalar weights are learnable nn.Parameters updated by the optimizer
    alongside the network. Softmax ensures they always sum to 1.

    Requires:  pip install pytorch-msssim
    Falls back to 3-component loss (no MS-SSIM) if package is unavailable.
    """

    def __init__(self, loss_weight=1.0):
        super(MultiscaleLoss, self).__init__()
        self.loss_weight = loss_weight

        self.charbonnier = CharbonnierLoss()
        self.perceptual  = VGGPerceptualLoss()
        self.gradient    = GradientLoss()
        self.softmax     = nn.Softmax(dim=1)

        try:
            from pytorch_msssim import MS_SSIM
            self.ms_ssim  = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1,
                                    size_average=True, channel=3)
            self.weights  = nn.Parameter(torch.rand(1, 4))  # 4 learnable weights
            self._n       = 4
        except ImportError:
            self.ms_ssim  = None
            self.weights  = nn.Parameter(torch.rand(1, 3))  # 3 learnable weights
            self._n       = 3

    def forward(self, pred, target):
        w = self.softmax(self.weights)   # (1, N), sums to 1

        l_charb = self.charbonnier(pred, target)
        l_per   = self.perceptual(pred, target)
        l_grad  = self.gradient(pred, target)

        if self.ms_ssim is not None:
            l_ms = 1.0 - self.ms_ssim(pred, target)   # higher MS-SSIM → lower loss
            loss = (w[0, 0] * l_charb + w[0, 1] * l_per +
                    w[0, 2] * l_grad  + w[0, 3] * l_ms)
        else:
            loss = w[0, 0] * l_charb + w[0, 1] * l_per + w[0, 2] * l_grad

        return self.loss_weight * loss


