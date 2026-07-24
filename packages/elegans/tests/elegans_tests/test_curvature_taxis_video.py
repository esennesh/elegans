"""Tests for reproducible curvature-taxis video rendering."""

from dataclasses import replace

import pytest
from elegans import curvature_taxis
from elegans.curvature_taxis import (
    CurvatureTaxisConfig,
    render_taxis_video,
    run_taxis,
    save_demo_artifacts,
)
from PIL import Image


@pytest.mark.parametrize(
    ("suffix", "kwargs", "match"),
    [
        (".avi", {}, r"\.mp4 or \.gif"),
        (".gif", {"fps": 0}, "positive integer"),
        (".gif", {"fps": True}, "positive integer"),
        (".gif", {"playback_speed": 0.0}, "greater than zero"),
        (".gif", {"playback_speed": float("nan")}, "finite"),
        (".gif", {"dpi": 0}, "positive integer"),
    ],
)
def test_video_renderer_rejects_invalid_output_settings(tmp_path, suffix, kwargs, match):
    """Invalid encodings and timing values fail before simulation or rendering."""
    with pytest.raises(ValueError, match=match):
        render_taxis_video(
            tmp_path / f"agent{suffix}",
            CurvatureTaxisConfig(),
            **kwargs,
        )


def test_short_trace_renders_as_an_animated_gif(tmp_path):
    """The portable writer produces multiple real trajectory frames."""
    config = replace(CurvatureTaxisConfig(), max_duration=0.12)
    trace = run_taxis(config)
    path = tmp_path / "agent.gif"

    rendered = render_taxis_video(
        path,
        config,
        trace,
        fps=5,
        playback_speed=0.2,
        dpi=40,
    )

    assert rendered == path
    assert path.stat().st_size > 1_000
    with Image.open(path) as image:
        assert image.size == (512, 288)
        assert getattr(image, "is_animated", False)
        assert getattr(image, "n_frames", 1) >= 2


def test_artifact_bundle_records_opt_in_video(monkeypatch, tmp_path):
    """The normal artifact pipeline passes its adaptive trace to the video renderer."""
    observed = {}

    def fake_renderer(path, config, trace, **kwargs):
        observed["path"] = path
        observed["config"] = config
        observed["trace"] = trace
        observed["kwargs"] = kwargs
        path.write_bytes(b"fake video")
        return path

    monkeypatch.setattr(curvature_taxis, "render_taxis_video", fake_renderer)
    config = replace(CurvatureTaxisConfig(), max_duration=0.12)

    summary = save_demo_artifacts(
        tmp_path,
        config,
        heading_count=1,
        render_video=True,
        video_fps=12,
        video_playback_speed=3.0,
    )

    video_path = tmp_path / "curvature_taxis_agent.mp4"
    assert video_path.read_bytes() == b"fake video"
    assert summary["artifacts"]["video"] == str(video_path)
    assert observed["path"] == video_path
    assert observed["config"] is config
    assert observed["trace"].speed_policy == "adaptive"
    assert observed["kwargs"] == {"fps": 12, "playback_speed": 3.0}
