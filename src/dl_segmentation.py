from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage import measure, morphology

from src.data_loader import CellTrackingSequence, find_dataset_root
from src.models.unet import UNet
from src.preprocess import prepare_frame_for_segmentation
from src.segmentation import DETECTION_COLUMNS


def _resolve_device(device: str | None = None) -> torch.device:
    """Resolve a requested inference device, falling back to CPU if needed."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")

    return resolved_device


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """Load a PyTorch checkpoint on the selected device."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}.")

    return checkpoint


def load_unet_model(
    checkpoint_path: Path,
    device: str | None = None,
) -> tuple[UNet, torch.device]:
    """Load a trained U-Net checkpoint for inference."""
    resolved_device = _resolve_device(device)
    checkpoint = _load_checkpoint(checkpoint_path, resolved_device)

    model = UNet(in_channels=1, out_channels=1, base_channels=32).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, resolved_device


def predict_mask(
    model: UNet,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """Predict a binary segmentation mask for one raw 2D frame or 3D stack."""
    prepared_image = prepare_frame_for_segmentation(image)
    image_tensor = (
        torch.from_numpy(np.ascontiguousarray(prepared_image))
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        mask = probabilities >= threshold

    return mask.squeeze(0).squeeze(0).cpu().numpy().astype(bool, copy=False)


def detections_from_mask(
    mask: np.ndarray,
    frame_index: int,
    min_size: int = 20,
) -> list[dict]:
    """Convert a binary mask to tracking-compatible detection dictionaries."""
    if min_size < 1:
        raise ValueError("min_size must be at least 1.")

    binary_mask = np.asarray(mask, dtype=bool)
    footprint = morphology.disk(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
        cleaned_mask = morphology.binary_opening(cleaned_mask, footprint=footprint)
        cleaned_mask = morphology.binary_closing(cleaned_mask, footprint=footprint)

    label_image = measure.label(cleaned_mask)
    detections: list[dict] = []

    for region in measure.regionprops(label_image):
        min_row, min_col, max_row, max_col = region.bbox
        centroid_y, centroid_x = region.centroid
        detection = {
            "frame": int(frame_index),
            "detection_id": int(region.label),
            "centroid_y": float(centroid_y),
            "centroid_x": float(centroid_x),
            "area": int(region.area),
            "bbox_min_row": int(min_row),
            "bbox_min_col": int(min_col),
            "bbox_max_row": int(max_row),
            "bbox_max_col": int(max_col),
        }
        detections.append({column: detection[column] for column in DETECTION_COLUMNS})

    return detections


def segment_frame_unet(
    image: np.ndarray,
    frame_index: int,
    model: UNet,
    device: torch.device,
    threshold: float = 0.5,
    min_size: int = 20,
) -> list[dict]:
    """Segment one microscopy frame with a trained U-Net."""
    mask = predict_mask(
        model=model,
        image=image,
        device=device,
        threshold=threshold,
    )
    return detections_from_mask(mask=mask, frame_index=frame_index, min_size=min_size)


def main() -> None:
    """Run U-Net inference on Fluo-N3DH-CHO sequence 01 frame 0."""
    checkpoint_path = Path("outputs") / "models" / "best_unet.pt"
    model, device = load_unet_model(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path, device)

    dataset_root = find_dataset_root("Fluo-N3DH-CHO")
    sequence = CellTrackingSequence(dataset_root, sequence="01")
    frame_index, image, path = sequence[0]

    detections = segment_frame_unet(
        image=image,
        frame_index=frame_index,
        model=model,
        device=device,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    if "val_dice" in checkpoint:
        print(f"val_dice: {checkpoint['val_dice']}")
    print(f"Frame: {path}")
    print(f"Detection count: {len(detections)}")
    print(f"First 5 detections: {detections[:5]}")


if __name__ == "__main__":
    main()
