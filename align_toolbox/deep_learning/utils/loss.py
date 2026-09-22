import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for binary segmentation with class-imbalance handling.

    Combines the Tversky index (a generalization of Dice that independently
    weights false positives and false negatives) with a focal exponent to
    down-weight easy examples and focus training on hard ones.

    Reference: Abraham & Khan (2019), arXiv:1810.07842.
    """

    def __init__(
        self,
        ignore_index=-1,
        smooth=100,
        alpha=0.3,
        beta=0.7,
        gamma=4 / 3,
        activation=True,
    ):
        """
        Parameters:
            ignore_index (int, optional): Target value to ignore in the loss
                computation. (default: -1)
            smooth (float, optional): Smoothing constant added to numerator and
                denominator of the Tversky index to avoid division by zero.
                (default: 100)
            alpha (float, optional): Weight for false positives in the Tversky
                index. (default: 0.3)
            beta (float, optional): Weight for false negatives in the Tversky
                index. (default: 0.7)
            gamma (float, optional): Focal exponent applied to ``(1 - Tversky)``.
                (default: 4/3)
            activation (bool, optional): If ``True``, apply sigmoid to inputs
                before computing the loss. (default: True)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.activation = activation

    def forward(self, inputs, targets):
        if self.activation:
            inputs = torch.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        smooth = self.smooth
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        # remove elements with ignore_index from the loss calculation
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky + 1e-10) ** gamma

        return FocalTversky


# adapted from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
class MultiClassFocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.

    Extends cross-entropy with a modulating factor ``(1 - p_t)^gamma`` that
    down-weights well-classified examples, focusing training on hard negatives.
    Supports per-class weights and an ignore index.

    Reference: Lin et al. (2017), arXiv:1708.02002.
    """

    def __init__(
        self,
        alpha: Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -1,
    ):
        """
        Parameters:
            alpha (Tensor, optional): Per-class weight tensor of shape
                ``(n_classes,)``. (default: None)
            gamma (float, optional): Focusing parameter; 0 recovers standard
                cross-entropy. (default: 2.0)
            reduction (str, optional): Reduction to apply: ``"mean"``, ``"sum"``,
                or ``"none"``. (default: ``"mean"``)
            ignore_index (int, optional): Target value to ignore. (default: -1)

        Raises:
            ValueError: If ``reduction`` is not one of the supported values.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0)
        x = x[unignored_mask]

        # get probabilities for focal term calculation
        log_p = F.log_softmax(x, dim=-1)

        ce = self.ce_loss(log_p, y)

        y = y.long()
        # get true class column from each row
        log_pt = torch.gather(log_p, 1, y.unsqueeze(1)).squeeze(1)

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class BCELossWithIgnore(nn.Module):
    """
    Binary cross-entropy loss that ignores a specified target value.

    Computes element-wise BCE loss, zeroes out entries where
    ``target == ignore_index``, and returns the mean over non-ignored elements.
    """

    def __init__(self, ignore_index=-1):
        """
        Parameters:
            ignore_index (int, optional): Target value to exclude from the loss.
                (default: -1)
        """
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction="none")
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # Calculate BCE loss

        # Create a mask for non-ignored values
        mask = (target != self.ignore_index).float()

        # Set the ignored values to 0
        target = target * mask

        # Calculate the loss
        loss = self.bce_loss(input, target)

        # Apply the mask to the loss
        masked_loss = loss * mask

        # Return the mean of the masked loss
        return masked_loss.sum() / mask.sum().clamp(min=1e-8)


class PeakWeightedMSELoss(nn.Module):
    """
    MSE loss with additional weight on high-value (peak) target positions.

    Assigns a per-element weight of ``1 + peak_weight * target`` so that
    positions with larger target values (peaks in a heatmap) contribute more to
    the loss. Designed for 1D keypoint heatmap regression.
    """

    def __init__(self, ignore_index=-1, peak_weight=3.0):
        """
        Parameters:
            ignore_index (int, optional): Reserved for API consistency; currently
                unused in the forward pass. (default: -1)
            peak_weight (float, optional): Additional weight multiplier for
                high-value target positions. (default: 3.0)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.peak_weight = peak_weight

    def forward(self, input, target):
        weights = 1.0 + self.peak_weight * target  # Higher weight for higher values
        return (weights * (input - target) ** 2).mean()


class MoltDetectionLoss(nn.Module):
    """
    Combined loss for molt detection, integrating heatmap regression and presence classification.

    This loss function combines a peak-weighted MSE loss for heatmap regression
    with a binary cross-entropy loss for presence classification. It is designed
    for 1D keypoint detection tasks where both the location of keypoints and their
    presence/absence need to be predicted.
    """

    def __init__(self, ignore_index=-1, peak_weight=3.0):
        """
        Parameters:
            ignore_index (int, optional): Reserved for API consistency; currently
                unused in the forward pass. (default: -1)
            peak_weight (float, optional): Additional weight multiplier for
                high-value target positions in the heatmap loss. (default: 3.0)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.peak_weight = peak_weight
        self.presence_loss_fn = F.binary_cross_entropy

    def heatmap_loss(self, predicted_heatmap, target_heatmap, mask):
        """
        Compute the heatmap regression loss.

        Parameters:
            predicted_heatmap (Tensor): Predicted heatmaps of shape [B, C, T].
            target_heatmap (Tensor): Ground truth heatmaps of shape [B, C, T].
            mask (Tensor): Boolean mask indicating valid (unpadded) frames of shape [B, 1, T].

        Returns:
            Tensor: The heatmap regression loss value.
        """
        weight = 1.0 + self.peak_weight * torch.maximum(
            target_heatmap, predicted_heatmap.detach()
        )
        sq = (predicted_heatmap - target_heatmap) ** 2
        loss = (weight * sq * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

    def forward(
        self,
        valid_mask,
        predicted_heatmap,
        predicted_presence,
        target_heatmap,
        target_presence,
    ):
        """
        Compute the combined loss.

        Parameters:
            valid_mask (Tensor): Boolean mask indicating valid (unpadded) frames of shape [B, T].
            predicted_heatmap (Tensor): Predicted heatmaps of shape [B, C, T].
            predicted_presence (Tensor): Predicted presence logits of shape [B, C].
            target_heatmap (Tensor): Ground truth heatmaps of shape [B, C, T].
            target_presence (Tensor): Ground truth presence labels of shape [B, C].

        Returns:
            Tensor: The combined loss value.
        """

        B, C, T = predicted_heatmap.shape

        dtype = predicted_heatmap.dtype

        mask = valid_mask.to(dtype)
        target_presence = target_presence.to(dtype)
        frame_mask = mask.view(B, 1, T)  # broadcast over channels

        heatmap_loss = self.heatmap_loss(predicted_heatmap, target_heatmap, frame_mask)
        presence_loss = self.presence_loss_fn(predicted_presence, target_presence)
        return heatmap_loss + presence_loss
