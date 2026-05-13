from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


def find_dataset_root() -> Path:
    candidates = [
        Path("data/raw/Fluo-N3DH-CHO/Fluo-N3DH-CHO"),
        Path("data/raw/Fluo-N3DH-CHO"),
    ]

    for candidate in candidates:
        if (candidate / "01").exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find dataset root. Expected a folder containing sequence folder '01'."
    )


def main() -> None:
    dataset_root = find_dataset_root()
    seq_dir = dataset_root / "01"

    tif_files = sorted(seq_dir.glob("*.tif")) + sorted(seq_dir.glob("*.tiff"))

    if not tif_files:
        raise FileNotFoundError(f"No tif files found in {seq_dir}")

    print(f"Dataset root: {dataset_root}")
    print(f"Sequence folder: {seq_dir}")
    print(f"Number of frames: {len(tif_files)}")

    first_file = tif_files[0]
    image = tiff.imread(first_file)

    print(f"First file: {first_file}")
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image min: {image.min()}")
    print(f"Image max: {image.max()}")

    if image.ndim == 3:
        preview = np.max(image, axis=0)
        title = f"{first_file.name} - MIP preview"
    else:
        preview = image
        title = first_file.name

    out_dir = Path("report/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.imshow(preview, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_dir / "first_frame_preview.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved preview to report/figures/first_frame_preview.png")


if __name__ == "__main__":
    main()
