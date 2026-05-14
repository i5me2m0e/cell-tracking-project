from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import find_dataset_root
from src.dl_dataset import CTCSegmentationDataset
from src.metrics import dice_score_from_logits
from src.models.unet import UNet


DEFAULT_DATASET_NAME = "Fluo-N3DH-CHO"
DEFAULT_OUTPUT = Path("outputs") / "models" / "best_unet.pt"
DatasetItem = tuple[torch.Tensor, torch.Tensor, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for U-Net training."""
    parser = argparse.ArgumentParser(
        description="Train a 2D U-Net for Fluo-N3DH-CHO binary segmentation."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training and validation batch size.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples used for validation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or a torch device string.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the best-model checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on invalid training arguments."""
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")


def set_seed(seed: int) -> None:
    """Seed the random number generators used by the training script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested command-line device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")

    return device


def build_datasets(
    dataset_root: Path,
    val_ratio: float,
    seed: int,
) -> tuple[Subset[DatasetItem], Subset[DatasetItem]]:
    """Build train/validation subsets with augmentation disabled for validation."""
    train_dataset: Dataset[DatasetItem] = CTCSegmentationDataset(
        dataset_root,
        sequences=("01", "02"),
        mask_source="ST",
        augment=True,
    )
    val_dataset: Dataset[DatasetItem] = CTCSegmentationDataset(
        dataset_root,
        sequences=("01", "02"),
        mask_source="ST",
        augment=False,
    )

    if len(train_dataset) != len(val_dataset):
        raise RuntimeError("Train and validation datasets do not have the same length.")

    num_samples = len(train_dataset)
    if num_samples < 2:
        raise ValueError("At least two paired image/mask samples are needed.")

    val_size = max(1, int(round(num_samples * val_ratio)))
    val_size = min(val_size, num_samples - 1)
    train_size = num_samples - val_size

    generator = torch.Generator().manual_seed(seed)
    train_indices_subset, val_indices_subset = random_split(
        range(num_samples),
        [train_size, val_size],
        generator=generator,
    )
    train_indices = [int(train_indices_subset[index]) for index in range(train_size)]
    val_indices = [int(val_indices_subset[index]) for index in range(val_size)]

    return Subset(train_dataset, train_indices), Subset(val_dataset, val_indices)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[DatasetItem],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return the sample-averaged loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, masks, _meta in loader:
        images = images.to(device=device, non_blocking=True)
        masks = masks.to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def validate(
    model: nn.Module,
    loader: DataLoader[DatasetItem],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation and return sample-averaged loss and Dice score."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks, _meta in loader:
            images = images.to(device=device, non_blocking=True)
            masks = masks.to(device=device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)
            dice = dice_score_from_logits(logits, masks)

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_dice += dice * batch_size
            total_samples += batch_size

    return total_loss / total_samples, total_dice / total_samples


def args_to_checkpoint_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Convert parsed args to a checkpoint-friendly dictionary."""
    args_dict = vars(args).copy()
    args_dict["output"] = str(args.output)
    return args_dict


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    epoch: int,
    val_dice: float,
    args: argparse.Namespace,
) -> None:
    """Save the best-model checkpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_dice": val_dice,
        "args": args_to_checkpoint_dict(args),
    }
    torch.save(checkpoint, output_path)


def run_training(args: argparse.Namespace) -> None:
    """Train U-Net and save the checkpoint with the best validation Dice."""
    validate_args(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    dataset_root = find_dataset_root(DEFAULT_DATASET_NAME)
    train_dataset, val_dataset = build_datasets(
        dataset_root=dataset_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    pin_memory = device.type == "cuda"
    train_loader: DataLoader[DatasetItem] = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader: DataLoader[DatasetItem] = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Dataset root: {dataset_root}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    best_val_dice = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_dice = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_dice={val_dice:.6f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(
                output_path=args.output,
                model=model,
                epoch=epoch,
                val_dice=val_dice,
                args=args,
            )

    print(f"Best val_dice={best_val_dice:.6f}")
    print(f"Saved best checkpoint to {args.output}")


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
