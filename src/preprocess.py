from __future__ import annotations

import numpy as np

from src.data_loader import CellTrackingSequence, find_dataset_root


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an image to [0, 1] using robust 1st/99th percentiles.

    The input is converted to float32 before normalization. Values below the
    1st percentile are clipped to 0, and values above the 99th percentile are
    clipped to 1. If the percentile range is invalid, a zero-filled float32
    image with the same shape is returned to avoid division by zero.
    """
    image_float = image.astype(np.float32, copy=False)
    p1, p99 = np.percentile(image_float, [1, 99])

    if p99 <= p1:
        return np.zeros_like(image_float, dtype=np.float32)

    normalized = (image_float - p1) / (p99 - p1)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def maximum_intensity_projection(image: np.ndarray) -> np.ndarray:
    """Return a 2D maximum intensity projection for a 3D image stack.

    A 3D image is projected along the first axis, interpreted as the z axis.
    A 2D image is returned unchanged. Images with any other dimensionality are
    rejected because there is no unambiguous 2D frame to segment.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return np.max(image, axis=0)

    raise ValueError(f"Expected a 2D or 3D image, got {image.ndim}D.")


def prepare_frame_for_segmentation(image: np.ndarray) -> np.ndarray:
    """Prepare one microscopy frame for 2D segmentation.

    3D z-stacks are first reduced to a 2D maximum intensity projection. The
    resulting 2D frame is then robustly normalized to a float32 image in [0, 1].
    """
    projected = maximum_intensity_projection(image)
    return normalize_image(projected)


def _print_image_stats(label: str, image: np.ndarray) -> None:
    """Print compact shape, dtype, and intensity statistics for an image."""
    print(
        f"{label} shape={image.shape} dtype={image.dtype} "
        f"min={image.min()} max={image.max()}"
    )


def main() -> None:
    """Load the first Fluo-N3DH-CHO/01 frame and print preprocessing stats."""
    dataset_root = find_dataset_root("Fluo-N3DH-CHO")
    sequence = CellTrackingSequence(dataset_root, sequence="01")
    _, raw_image, path = sequence[0]
    processed_image = prepare_frame_for_segmentation(raw_image)

    print(f"Frame: {path}")
    _print_image_stats("raw", raw_image)
    _print_image_stats("processed", processed_image)


if __name__ == "__main__":
    main()
