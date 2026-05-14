from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, TypedDict

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from src.data_loader import find_dataset_root
from src.preprocess import maximum_intensity_projection, normalize_image


_IMAGE_RE = re.compile(r"^t(\d+)\.tiff?$", re.IGNORECASE)
_MASK_RE = re.compile(r"^man_seg(\d+)\.tiff?$", re.IGNORECASE)
_MASK_WITH_SLICE_RE = re.compile(r"^man_seg_(\d+)(?:_\d+)?\.tiff?$", re.IGNORECASE)


class CTCSegmentationSample(TypedDict):
    sequence: str
    frame_index: int
    image_path: Path
    mask_path: Path


class CTCSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    """Dataset for paired CTC raw frames and segmentation masks."""

    def __init__(
        self,
        dataset_root: Path,
        sequences: tuple[str, ...] = ("01", "02"),
        mask_source: str = "ST",
        augment: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.sequences = sequences
        self.mask_source = mask_source.upper()
        self.augment = augment

        if self.mask_source not in {"ST", "GT"}:
            raise ValueError("mask_source must be either 'ST' or 'GT'.")

        self.samples = self._build_samples()

    def _build_samples(self) -> list[CTCSegmentationSample]:
        samples: list[CTCSegmentationSample] = []

        for sequence in self.sequences:
            image_dir = self.dataset_root / sequence
            mask_dir = self.dataset_root / f"{sequence}_{self.mask_source}" / "SEG"

            if not image_dir.is_dir():
                raise FileNotFoundError(f"Image sequence directory not found: {image_dir}")
            if not mask_dir.is_dir():
                raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

            image_paths = self._index_paths_by_frame(image_dir, _parse_image_frame_index)
            mask_paths = self._index_paths_by_frame(mask_dir, _parse_mask_frame_index)

            for frame_index in sorted(image_paths.keys() & mask_paths.keys()):
                samples.append(
                    {
                        "sequence": sequence,
                        "frame_index": frame_index,
                        "image_path": image_paths[frame_index],
                        "mask_path": mask_paths[frame_index],
                    }
                )

        return samples

    @staticmethod
    def _index_paths_by_frame(
        directory: Path,
        parse_frame_index: Callable[[Path], int | None],
    ) -> dict[int, Path]:
        paths_by_frame: dict[int, Path] = {}

        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
                continue

            frame_index = parse_frame_index(path)
            if frame_index is None or frame_index in paths_by_frame:
                continue

            paths_by_frame[frame_index] = path

        return paths_by_frame

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]

        image = tifffile.imread(sample["image_path"])
        image = maximum_intensity_projection(image)
        image = normalize_image(image)

        mask = tifffile.imread(sample["mask_path"])
        mask = maximum_intensity_projection(mask)
        mask = (mask > 0).astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)

        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .to(dtype=torch.float32)
            .unsqueeze(0)
        )
        mask_tensor = (
            torch.from_numpy(np.ascontiguousarray(mask))
            .to(dtype=torch.float32)
            .unsqueeze(0)
        )
        meta: dict[str, Any] = {
            "sequence": sample["sequence"],
            "frame_index": sample["frame_index"],
            "image_path": str(sample["image_path"]),
            "mask_path": str(sample["mask_path"]),
        }

        return image_tensor, mask_tensor, meta

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if torch.rand(()).item() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        if torch.rand(()).item() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)

        return image, mask


def _parse_image_frame_index(path: Path) -> int | None:
    match = _IMAGE_RE.match(path.name)
    if match is None:
        return None

    return int(match.group(1))


def _parse_mask_frame_index(path: Path) -> int | None:
    match = _MASK_RE.match(path.name)
    if match is not None:
        return int(match.group(1))

    match = _MASK_WITH_SLICE_RE.match(path.name)
    if match is not None:
        return int(match.group(1))

    return None


def _tensor_stats(label: str, tensor: torch.Tensor) -> None:
    print(
        f"{label} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"min={tensor.min().item()} max={tensor.max().item()}"
    )


def main() -> None:
    dataset_root = find_dataset_root("Fluo-N3DH-CHO")
    dataset = CTCSegmentationDataset(dataset_root)

    print(f"Dataset root: {dataset_root}")
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) == 0:
        return

    image, mask, meta = dataset[0]
    _tensor_stats("image", image)
    _tensor_stats("mask", mask)
    print(f"meta: {meta}")


if __name__ == "__main__":
    main()
