from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas
from scipy.optimize import linear_sum_assignment


TRACK_COLUMNS = [
    "frame",
    "track_id",
    "detection_id",
    "centroid_y",
    "centroid_x",
    "area",
]

REQUIRED_DETECTION_COLUMNS = [
    "frame",
    "detection_id",
    "centroid_y",
    "centroid_x",
    "area",
]


@dataclass
class ActiveTrack:
    """State needed to link the next frame to an existing track."""

    track_id: int
    last_frame: int
    centroid_y: float
    centroid_x: float


def _check_detection_columns(detections_df: pandas.DataFrame) -> None:
    missing_columns = [
        column
        for column in REQUIRED_DETECTION_COLUMNS
        if column not in detections_df.columns
    ]
    if missing_columns:
        raise ValueError(f"detections_df is missing columns: {missing_columns}")


def _row_to_track_point(row: pandas.Series, track_id: int) -> dict:
    return {
        "frame": int(row["frame"]),
        "track_id": int(track_id),
        "detection_id": int(row["detection_id"]),
        "centroid_y": float(row["centroid_y"]),
        "centroid_x": float(row["centroid_x"]),
        "area": float(row["area"]),
    }


def link_detections(
    detections_df: pandas.DataFrame,
    max_distance: float = 30.0,
    max_missing: int = 2,
) -> pandas.DataFrame:
    """Link per-frame detections into tracks with Hungarian matching.

    Each active track is represented by its most recent centroid. For every new
    frame, detections are matched to active tracks by minimum total Euclidean
    distance. Matches farther than ``max_distance`` are rejected, and unmatched
    detections start new tracks.
    """
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative.")
    if max_missing < 0:
        raise ValueError("max_missing must be non-negative.")

    _check_detection_columns(detections_df)

    if detections_df.empty:
        return pandas.DataFrame(columns=TRACK_COLUMNS)

    sorted_detections = detections_df.sort_values(
        ["frame", "detection_id"],
        kind="mergesort",
    )

    active_tracks: dict[int, ActiveTrack] = {}
    track_points: list[dict] = []
    next_track_id = 0

    for frame, frame_detections in sorted_detections.groupby("frame", sort=True):
        current_frame = int(frame)

        active_tracks = {
            track_id: track
            for track_id, track in active_tracks.items()
            if current_frame - track.last_frame - 1 <= max_missing
        }

        frame_detections = frame_detections.reset_index(drop=True)
        matched_detection_indices: set[int] = set()

        if active_tracks:
            track_ids = list(active_tracks.keys())
            previous_positions = np.array(
                [
                    [
                        active_tracks[track_id].centroid_y,
                        active_tracks[track_id].centroid_x,
                    ]
                    for track_id in track_ids
                ],
                dtype=float,
            )
            current_positions = frame_detections[["centroid_y", "centroid_x"]].to_numpy(
                dtype=float,
            )
            distance_matrix = np.linalg.norm(
                previous_positions[:, np.newaxis, :]
                - current_positions[np.newaxis, :, :],
                axis=2,
            )

            matched_track_rows, matched_detection_columns = linear_sum_assignment(
                distance_matrix
            )

            for track_row, detection_column in zip(
                matched_track_rows,
                matched_detection_columns,
            ):
                distance = distance_matrix[track_row, detection_column]
                if distance > max_distance:
                    continue

                track_id = track_ids[track_row]
                detection = frame_detections.iloc[detection_column]
                track_points.append(_row_to_track_point(detection, track_id))
                active_tracks[track_id] = ActiveTrack(
                    track_id=track_id,
                    last_frame=current_frame,
                    centroid_y=float(detection["centroid_y"]),
                    centroid_x=float(detection["centroid_x"]),
                )
                matched_detection_indices.add(int(detection_column))

        for detection_index, detection in frame_detections.iterrows():
            if int(detection_index) in matched_detection_indices:
                continue

            track_id = next_track_id
            next_track_id += 1
            track_points.append(_row_to_track_point(detection, track_id))
            active_tracks[track_id] = ActiveTrack(
                track_id=track_id,
                last_frame=current_frame,
                centroid_y=float(detection["centroid_y"]),
                centroid_x=float(detection["centroid_x"]),
            )

    return pandas.DataFrame(track_points, columns=TRACK_COLUMNS)


def summarize_tracks(tracks_df: pandas.DataFrame) -> dict:
    """Return simple track-count and track-length statistics."""
    if tracks_df.empty:
        return {
            "num_tracks": 0,
            "num_track_points": 0,
            "mean_track_length": 0.0,
            "max_track_length": 0,
        }

    if "track_id" not in tracks_df.columns:
        raise ValueError("tracks_df is missing column: track_id")

    track_lengths = tracks_df.groupby("track_id").size()
    return {
        "num_tracks": int(track_lengths.shape[0]),
        "num_track_points": int(len(tracks_df)),
        "mean_track_length": float(track_lengths.mean()),
        "max_track_length": int(track_lengths.max()),
    }


def main() -> None:
    """Run a small self-contained test without reading any dataset files."""
    detections_df = pandas.DataFrame(
        [
            {
                "frame": 0,
                "detection_id": 1,
                "centroid_y": 10.0,
                "centroid_x": 10.0,
                "area": 50,
            },
            {
                "frame": 0,
                "detection_id": 2,
                "centroid_y": 80.0,
                "centroid_x": 80.0,
                "area": 70,
            },
            {
                "frame": 1,
                "detection_id": 1,
                "centroid_y": 12.0,
                "centroid_x": 11.0,
                "area": 52,
            },
            {
                "frame": 1,
                "detection_id": 2,
                "centroid_y": 79.0,
                "centroid_x": 82.0,
                "area": 68,
            },
            {
                "frame": 1,
                "detection_id": 3,
                "centroid_y": 150.0,
                "centroid_x": 150.0,
                "area": 40,
            },
            {
                "frame": 3,
                "detection_id": 1,
                "centroid_y": 15.0,
                "centroid_x": 13.0,
                "area": 55,
            },
            {
                "frame": 4,
                "detection_id": 1,
                "centroid_y": 152.0,
                "centroid_x": 149.0,
                "area": 42,
            },
        ]
    )

    tracks_df = link_detections(detections_df, max_distance=10.0, max_missing=2)
    print(tracks_df)
    print(summarize_tracks(tracks_df))


if __name__ == "__main__":
    main()
