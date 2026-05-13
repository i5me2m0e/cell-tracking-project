from __future__ import annotations

import warnings

import numpy as np
import pandas
from skimage import filters, measure, morphology

from src.data_loader import CellTrackingSequence, find_dataset_root
from src.preprocess import prepare_frame_for_segmentation


DETECTION_COLUMNS = [
    "frame",
    "detection_id",
    "centroid_y",
    "centroid_x",
    "area",
    "bbox_min_row",
    "bbox_min_col",
    "bbox_max_row",
    "bbox_max_col",
]


def segment_frame(
    image: np.ndarray,
    frame_index: int,
    min_size: int = 20,
    gaussian_sigma: float = 1.0,
) -> list[dict]:
    """Segment one microscopy frame with a classical image-processing baseline.

    The input frame may be either 2D or a 3D z-stack. It is first converted to a
    normalized 2D image by ``prepare_frame_for_segmentation``. Bright foreground
    regions are then detected with Gaussian denoising, Otsu thresholding,
    morphology cleanup, connected-component labeling, and region properties.
    """
    if min_size < 1:
        raise ValueError("min_size must be at least 1.")
    if gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be non-negative.")

    prepared_image = prepare_frame_for_segmentation(image)
    blurred_image = filters.gaussian(
        prepared_image,
        sigma=gaussian_sigma,
        preserve_range=True,
    )

    threshold = filters.threshold_otsu(blurred_image)
    binary_mask = blurred_image > threshold

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
        detections.append(
            {
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
        )

    return detections


def detections_to_dataframe(detections: list[dict]) -> pandas.DataFrame:
    """Convert detection dictionaries to a pandas DataFrame."""
    return pandas.DataFrame(detections, columns=DETECTION_COLUMNS)


def main() -> None:
    """Run the baseline on Fluo-N3DH-CHO sequence 01 frame 0."""
    dataset_root = find_dataset_root("Fluo-N3DH-CHO")
    sequence = CellTrackingSequence(dataset_root, sequence="01")
    frame_index, image, path = sequence[0]

    detections = segment_frame(image, frame_index=frame_index)
    detections_df = detections_to_dataframe(detections)

    print(f"Frame: {path}")
    print(f"Detection count: {len(detections)}")
    print(detections_df.head(5))


if __name__ == "__main__":
    main()
