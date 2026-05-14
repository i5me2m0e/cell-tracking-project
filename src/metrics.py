from __future__ import annotations

import torch


def dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute the batch-averaged binary Dice score from logits."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    cardinality = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)

    return float(dice.mean().item())


def iou_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute the batch-averaged binary IoU score from logits."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
    iou = (intersection + eps) / (union + eps)

    return float(iou.mean().item())


def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute soft Dice loss from logits."""
    probs = torch.sigmoid(logits)
    targets = targets.float()

    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)

    return 1.0 - dice.mean()


def main() -> None:
    logits = torch.randn(4, 1, 64, 64)
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float()

    dice_score = dice_score_from_logits(logits, targets)
    iou_score = iou_score_from_logits(logits, targets)
    dice_loss = dice_loss_from_logits(logits, targets)

    print(f"dice score: {dice_score:.6f}")
    print(f"iou score: {iou_score:.6f}")
    print(f"dice loss: {dice_loss.item():.6f}")


if __name__ == "__main__":
    main()
