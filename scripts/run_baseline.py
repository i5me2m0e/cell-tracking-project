from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import CellTrackingSequence, find_dataset_root
from src.segmentation import detections_to_dataframe, segment_frame
from src.tracking import link_detections, summarize_tracks
from src.visualize import make_gif, save_tracking_frames


DEFAULT_DATASET_NAME = "Fluo-N3DH-CHO"
TRACK_OUTPUT_DIR = Path("outputs") / "tracks"
VIDEO_OUTPUT_DIR = Path("outputs") / "videos"
FRAME_OUTPUT_DIR = VIDEO_OUTPUT_DIR / "frames"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the baseline pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the classical segmentation and tracking baseline."
    )
    parser.add_argument("--sequence", default="01", help="Sequence folder to process.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=30.0,
        help="Maximum linking distance between detections.",
    )
    parser.add_argument(
        "--max-missing",
        type=int,
        default=2,
        help="Maximum number of missing frames allowed while keeping a track active.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=20,
        help="Minimum segmented object size in pixels.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma used before thresholding.",
    )
    parser.add_argument("--fps", type=int, default=5, help="Output GIF frame rate.")
    parser.add_argument(
        "--history",
        type=int,
        default=10,
        help="Number of recent frames to draw for each track.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on invalid numeric arguments."""
    if args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1.")
    if args.max_distance < 0:
        raise ValueError("--max-distance must be non-negative.")
    if args.max_missing < 0:
        raise ValueError("--max-missing must be non-negative.")
    if args.min_size < 1:
        raise ValueError("--min-size must be at least 1.")
    if args.gaussian_sigma < 0:
        raise ValueError("--gaussian-sigma must be non-negative.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.history < 1:
        raise ValueError("--history must be at least 1.")


def run_baseline(args: argparse.Namespace) -> None:
    """Run segmentation, tracking, summary reporting, and visualization."""
    validate_args(args)

    dataset_root = find_dataset_root(DEFAULT_DATASET_NAME)
    sequence = CellTrackingSequence(dataset_root, sequence=args.sequence)
    if len(sequence) == 0:
        raise ValueError(f"No frames found for sequence {args.sequence!r}.")

    max_frames = min(args.max_frames, len(sequence))

    print(f"Dataset root: {dataset_root}")
    print(f"Sequence: {args.sequence}")
    print(f"Frames processed: {max_frames} / {len(sequence)}")

    all_detections: list[dict] = []
    for frame_number, (frame_index, image, _path) in enumerate(sequence):
        if frame_number >= max_frames:
            break

        frame_detections = segment_frame(
            image=image,
            frame_index=frame_index,
            min_size=args.min_size,
            gaussian_sigma=args.gaussian_sigma,
        )
        all_detections.extend(frame_detections)

    TRACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detections_df = detections_to_dataframe(all_detections)
    detections_path = TRACK_OUTPUT_DIR / "detections.csv"
    detections_df.to_csv(detections_path, index=False)

    tracks_df = link_detections(
        detections_df=detections_df,
        max_distance=args.max_distance,
        max_missing=args.max_missing,
    )
    tracks_path = TRACK_OUTPUT_DIR / "tracks.csv"
    tracks_df.to_csv(tracks_path, index=False)

    summary = summarize_tracks(tracks_df)
    print("Summary:")
    print(f"  detections: {len(detections_df)}")
    print(f"  tracks: {summary['num_tracks']}")
    print(f"  track points: {summary['num_track_points']}")
    print(f"  mean track length: {summary['mean_track_length']:.2f}")
    print(f"  max track length: {summary['max_track_length']}")

    frame_paths = save_tracking_frames(
        sequence=sequence,
        tracks_df=tracks_df,
        output_dir=FRAME_OUTPUT_DIR,
        max_frames=max_frames,
        history=args.history,
    )
    gif_path = VIDEO_OUTPUT_DIR / "tracking_result.gif"
    make_gif(frame_paths, gif_path, fps=args.fps)

    print(f"Saved detections to {detections_path}")
    print(f"Saved tracks to {tracks_path}")
    print(f"Saved {len(frame_paths)} visualization frames to {FRAME_OUTPUT_DIR}")
    print(f"Saved GIF to {gif_path}")


def main() -> None:
    args = parse_args()
    run_baseline(args)


if __name__ == "__main__":
    main()
