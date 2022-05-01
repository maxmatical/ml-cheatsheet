from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(
        self, weight: Optional[torch.Tensor] = None, gamma: int = 2, reduction: str = "mean"
    ):
        """
        Version of (Weighted) focal loss for multi-class classification with CE loss
        Taken from https://arxiv.org/abs/1708.02002

        Args:
            weights (Optional, torch.Tensor): a tensor of alpha parameter to balance class weights
            gamma (int): focus parameter.
                higher gamma => more "easy" examples with low loss is discounted
                when gamma == 0, focal loss is equivalent to CE Loss
                defaults to 2 based on findings
            reduction (str): reduction strategy for CE loss

        """
        super().__init__(weight=weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class PolyCELoss(nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        epsilon: float = 1.0,
    ) -> None:
        """
        Pytorch implementation of Poly Cross Entropy Loss from
        https://arxiv.org/abs/2204.12511v1
        This version uses logits for input, don't use softmaxed input
        """
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.epsilon = epsilon

        if self.reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f'Unsupported reduction: {self.reduction},\
                available options are ["mean", "sum", "none"].'
            )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input must be logits, not probabilities
        """
        probs = torch.softmax(input, dim=1)
        pt = (target * probs).sum(dim=1)
        ce_loss = F.cross_entropy(
            input,
            target,
            reduction="none",
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )
        poly_loss = ce_loss + self.epsilon * (1 - pt)

        if self.reduction == "mean":
            poly_loss = poly_loss.mean()
        elif self.reduction == "sum":
            poly_loss = poly_loss.sum()
        else:
            poly_loss = poly_loss.unsqueeze(dim=1)
        return poly_loss


class PolyFocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: int = 2,
        reduction: str = "mean",
        epsilon: float = 1.0,
    ) -> None:
        """
        Pytorch implementation of Poly Focal Loss from
        https://arxiv.org/abs/2204.12511v1
        Adjusted for multiclass classification
        """
        super().__init__(weight=weight, reduction=reduction)
        self.focal_loss = FocalLoss(weight, gamma=gamma, reduction=reduction)
        self.epsilon = epsilon
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input must be logits, not probabilities
        """
        probs = torch.softmax(input, dim=1)
        pt = (target * probs).sum(dim=1)
        fl = self.focal_loss(input, target)
        poly_fl = fl + self.epsilon * (1 - pt) ** (self.gamma + 1)
        return poly_fl.mean()
