from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas

from src.data_loader import CellTrackingSequence, find_dataset_root
from src.preprocess import maximum_intensity_projection, normalize_image


REQUIRED_TRACK_COLUMNS = [
    "frame",
    "track_id",
    "detection_id",
    "centroid_y",
    "centroid_x",
    "area",
]

TRACK_COLORS_RGB = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
]


def image_to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a 2D frame or 3D stack to an RGB uint8 image."""
    if image.ndim == 3:
        image = maximum_intensity_projection(image)
    elif image.ndim != 2:
        raise ValueError(f"Expected a 2D image or 3D stack, got {image.ndim}D.")

    normalized = normalize_image(image)
    gray_uint8 = np.rint(normalized * 255.0).astype(np.uint8)
    return np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)


def draw_tracks_on_frame(
    image: np.ndarray,
    frame_index: int,
    tracks_df: pandas.DataFrame,
    history: int = 10,
) -> np.ndarray:
    """Draw current detections and recent track history on one frame."""
    _check_track_columns(tracks_df)

    output = image_to_uint8_rgb(image).copy()
    if tracks_df.empty:
        return output

    current_frame = int(frame_index)
    history = max(1, int(history))
    first_history_frame = current_frame - history + 1

    recent_tracks = tracks_df[
        (tracks_df["frame"] >= first_history_frame)
        & (tracks_df["frame"] <= current_frame)
    ]
    current_tracks = tracks_df[tracks_df["frame"] == current_frame]

    for track_id, track_points in recent_tracks.groupby("track_id", sort=True):
        sorted_points = track_points.sort_values("frame", kind="mergesort")
        points = [_row_to_point(row) for _, row in sorted_points.iterrows()]
        points = [point for point in points if point is not None]
        if len(points) < 2:
            continue

        color = _track_color(track_id)
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(output, start, end, color, thickness=2, lineType=cv2.LINE_AA)

    for _, row in current_tracks.iterrows():
        point = _row_to_point(row)
        if point is None:
            continue

        track_id = row["track_id"]
        color = _track_color(track_id)
        label = str(int(track_id))
        label_origin = (point[0] + 6, point[1] - 6)

        cv2.circle(output, point, radius=5, color=(0, 0, 0), thickness=1)
        cv2.circle(output, point, radius=4, color=color, thickness=-1)
        cv2.putText(
            output,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return output


def save_tracking_frames(
    sequence: CellTrackingSequence,
    tracks_df: pandas.DataFrame,
    output_dir: Path,
    max_frames: int | None = None,
    history: int = 10,
) -> list[Path]:
    """Draw and save tracking overlays for frames in a sequence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for frame_index, image, _path in sequence:
        if max_frames is not None and len(saved_paths) >= max_frames:
            break

        drawn_frame = draw_tracks_on_frame(
            image=image,
            frame_index=frame_index,
            tracks_df=tracks_df,
            history=history,
        )
        output_path = output_dir / f"frame_{frame_index:04d}.png"
        imageio.imwrite(output_path, drawn_frame)
        saved_paths.append(output_path)

    return saved_paths


def make_gif(frame_paths: list[Path], output_path: Path, fps: int = 5) -> None:
    """Read PNG frames and save them as a GIF."""
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if not frame_paths:
        raise ValueError("frame_paths must contain at least one frame.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [imageio.imread(Path(frame_path)) for frame_path in frame_paths]
    imageio.mimsave(output_path, frames, duration=1.0 / fps)


def _check_track_columns(tracks_df: pandas.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_TRACK_COLUMNS if column not in tracks_df.columns
    ]
    if missing_columns:
        raise ValueError(f"tracks_df is missing columns: {missing_columns}")


def _track_color(track_id: object) -> tuple[int, int, int]:
    color_index = int(track_id) % len(TRACK_COLORS_RGB)
    return TRACK_COLORS_RGB[color_index]


def _row_to_point(row: pandas.Series) -> tuple[int, int] | None:
    y = float(row["centroid_y"])
    x = float(row["centroid_x"])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return int(round(x)), int(round(y))


def _make_demo_tracks(image_shape: tuple[int, int], num_frames: int) -> pandas.DataFrame:
    height, width = image_shape
    rows: list[dict] = []

    base_tracks = [
        (0, 0.30, 0.30, 0.04, 0.05),
        (1, 0.62, 0.45, -0.03, 0.04),
        (2, 0.48, 0.70, 0.02, -0.05),
    ]

    detection_id = 0
    for frame in range(num_frames):
        for track_id, y0, x0, dy, dx in base_tracks:
            rows.append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "detection_id": detection_id,
                    "centroid_y": height * (y0 + dy * frame),
                    "centroid_x": width * (x0 + dx * frame),
                    "area": 80.0,
                }
            )
            detection_id += 1

    return pandas.DataFrame(rows, columns=REQUIRED_TRACK_COLUMNS)


def main() -> None:
    """Create a small tracking visualization on Fluo-N3DH-CHO sequence 01."""
    dataset_root = find_dataset_root("Fluo-N3DH-CHO")
    sequence = CellTrackingSequence(dataset_root, sequence="01")
    num_frames = min(3, len(sequence))

    _, first_image, _ = sequence[0]
    first_frame = maximum_intensity_projection(first_image)
    tracks_df = _make_demo_tracks(first_frame.shape, num_frames)

    frame_paths = save_tracking_frames(
        sequence=sequence,
        tracks_df=tracks_df,
        output_dir=Path("outputs/videos/test_frames"),
        max_frames=num_frames,
        history=3,
    )
    gif_path = Path("outputs/videos/test_tracking.gif")
    make_gif(frame_paths, gif_path, fps=5)

    print(f"Saved {len(frame_paths)} frames to outputs/videos/test_frames")
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    main()
