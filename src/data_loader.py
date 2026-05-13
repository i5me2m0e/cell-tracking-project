from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile as tiff


def find_dataset_root(dataset_name: str = "Fluo-N3DH-CHO") -> Path:
    """Find the Cell Tracking Challenge dataset root under data/raw."""
    base_dir = Path("data") / "raw" / dataset_name
    candidates = [
        base_dir / dataset_name,
        base_dir,
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "01").is_dir():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find dataset root. Searched: {searched}")


def list_sequence_frames(dataset_root: Path, sequence: str = "01") -> list[Path]:
    """Return all TIFF frames in a sequence directory sorted by filename."""
    sequence_dir = dataset_root / sequence
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")

    frames = [
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    ]
    return sorted(frames, key=lambda path: path.name)


def read_frame(path: Path) -> np.ndarray:
    """Read a single TIFF frame without changing its dimensionality."""
    return tiff.imread(path)


class CellTrackingSequence:
    """Simple indexable and iterable wrapper for one CTC image sequence."""

    def __init__(self, dataset_root: Path, sequence: str = "01") -> None:
        self.dataset_root = dataset_root
        self.sequence = sequence
        self.frames = list_sequence_frames(dataset_root, sequence)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> tuple[int, np.ndarray, Path]:
        path = self.frames[index]
        return index, read_frame(path), path

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, Path]]:
        for index in range(len(self)):
            yield self[index]


def main() -> None:
    dataset_root = find_dataset_root()
    sequence = CellTrackingSequence(dataset_root, sequence="01")

    print(f"Dataset root: {dataset_root}")
    print(f"Sequence: {sequence.sequence}")
    print(f"Number of frames: {len(sequence)}")

    for frame_index, image, path in sequence:
        if frame_index >= 3:
            break
        print(
            f"{frame_index}: {path.name} "
            f"shape={image.shape} dtype={image.dtype} "
            f"min={image.min()} max={image.max()}"
        )


if __name__ == "__main__":
    main()
