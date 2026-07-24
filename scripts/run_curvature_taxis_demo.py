"""Run local field-curvature sensing and curvature-controlled locomotion."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from elegans.curvature_taxis import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_PLAYBACK_SPEED,
    CurvatureTaxisConfig,
    save_demo_artifacts,
)

DEFAULT_HEADING_COUNT = 20


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/curvature_taxis"),
        help="Directory for the figure, traces, sweep, summary, and optional video.",
    )
    parser.add_argument(
        "--initial-heading-degrees",
        type=float,
        default=CurvatureTaxisConfig().initial_heading_degrees,
        help="Initial heading for the primary comparison (default: 28).",
    )
    parser.add_argument(
        "--heading-count",
        type=int,
        default=DEFAULT_HEADING_COUNT,
        help="Number of evenly spaced headings in the robustness sweep (default: 20).",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Also render curvature_taxis_agent.mp4 from the actual adaptive trace.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        help=f"Video frame rate (default: {DEFAULT_VIDEO_FPS}).",
    )
    parser.add_argument(
        "--video-playback-speed",
        type=float,
        default=DEFAULT_VIDEO_PLAYBACK_SPEED,
        help=(
            f"Ratio of simulated time to video time (default: {DEFAULT_VIDEO_PLAYBACK_SPEED:g}x)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the demo, save all outputs, and print its machine-readable summary."""
    args = parse_args()
    config = replace(
        CurvatureTaxisConfig(),
        initial_heading_degrees=args.initial_heading_degrees,
    )
    summary = save_demo_artifacts(
        args.output_dir,
        config,
        heading_count=args.heading_count,
        render_video=args.video,
        video_fps=args.video_fps,
        video_playback_speed=args.video_playback_speed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
