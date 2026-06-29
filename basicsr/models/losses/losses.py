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


class PhaseFormerMultiScaleLoss(nn.Module):
    """Multi-scale loss for 1x + 2x super-resolution outputs.
    
    Applies L1 loss to both 1x and 2x outputs.
    For 2x, the ground truth is upsampled using bilinear interpolation.
    
    Args:
        loss_weight (float): Overall loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        weight_1x (float): Weight for 1x loss. Default: 0.5.
        weight_2x (float): Weight for 2x loss. Default: 0.5.
    """
    
    def __init__(self, loss_weight=1.0, reduction='mean', weight_1x=0.5, weight_2x=0.5):
        super(PhaseFormerMultiScaleLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.weight_1x = weight_1x
        self.weight_2x = weight_2x
        
        # Ensure weights sum to 1 (normalize)
        total_weight = weight_1x + weight_2x
        self.weight_1x = weight_1x / total_weight if total_weight > 0 else 0.5
        self.weight_2x = weight_2x / total_weight if total_weight > 0 else 0.5
    
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (list of Tensors): [pred_1x, pred_2x] where:
                pred_1x: shape (N, C, H, W) - 1x resolution output
                pred_2x: shape (N, C, 2H, 2W) - 2x resolution output
            target (Tensor): shape (N, C, H, W) - ground truth at 1x resolution
            weight (Tensor, optional): not used for multi-scale
        """
        if not isinstance(pred, (list, tuple)) or len(pred) != 2:
            raise ValueError(f'Expected pred to be a list of 2 tensors [pred_1x, pred_2x], got {type(pred)}')
        
        pred_1x, pred_2x = pred
        
        # Loss for 1x branch
        loss_1x = l1_loss(pred_1x, target, weight, reduction=self.reduction)
        
        # Upsample target to 2x for 2x branch (bilinear interpolation)
        # Note: target is (N, C, H, W), we need to upscale to (N, C, 2H, 2W)
        _, _, h, w = target.shape
        target_2x = F.interpolate(target, size=(h*2, w*2), mode='bilinear', align_corners=False)
        
        # Loss for 2x branch
        loss_2x = l1_loss(pred_2x, target_2x, weight, reduction=self.reduction)
        
        # Weighted combination
        total_loss = self.loss_weight * (self.weight_1x * loss_1x + self.weight_2x * loss_2x)
        
        return total_loss
