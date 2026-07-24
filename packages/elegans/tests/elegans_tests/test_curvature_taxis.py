"""Tests for sensed field curvature and curvature-controlled speed."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest
from elegans.curvature_taxis import (
    CurvatureTaxisConfig,
    CurvatureTaxisEnvironment,
    TerminationReason,
    adaptive_speed,
    bilateral_turn_rate,
    run_heading_sweep,
    run_matched_constant_speed,
    run_taxis,
    save_demo_artifacts,
    summarize_heading_sweep,
)
from elegans.field_geometry import curvature_controlled_speed
from matplotlib import image as mpimg


def test_field_is_non_circular_non_gaussian_and_has_unique_peak():
    """The field is quartic, anisotropic, and maximized only at its source."""
    field = CurvatureTaxisConfig().field()
    source = np.asarray(field.source)
    direction = np.array([1.0, 0.0])
    perpendicular = np.array([0.0, 1.0])

    assert field.beta > 0.0
    assert field.gamma > 0.0
    assert field.concentration(source) == pytest.approx(1.0)
    assert field.concentration(source + 1.4 * direction) != pytest.approx(
        field.concentration(source + 1.4 * perpendicular),
    )
    for offset in ((0.2, 0.0), (-0.2, 0.1), (0.0, -0.2), (0.15, 0.15)):
        assert field.concentration(source + offset) < 1.0


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("field_beta", np.nan),
        ("field_beta", np.inf),
        ("field_gamma", np.nan),
        ("field_gamma", -np.inf),
        ("field_rotation_degrees", np.nan),
        ("field_rotation_degrees", np.inf),
    ],
)
def test_config_rejects_nonfinite_field_shape_parameters(field_name, invalid_value):
    """NaN and infinity cannot silently enter the field geometry."""
    with pytest.raises(ValueError, match="finite"):
        replace(CurvatureTaxisConfig(), **{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("beta", np.nan),
        ("gamma", np.inf),
        ("rotation_degrees", -np.inf),
        ("scale_u", np.nan),
    ],
)
def test_direct_field_construction_rejects_nonfinite_parameters(field_name, invalid_value):
    """The public field class enforces the same finite-value contract."""
    with pytest.raises(ValueError, match="finite"):
        replace(CurvatureTaxisConfig().field(), **{field_name: invalid_value})


def test_analytic_derivatives_match_finite_differences():
    """Closed-form gradient and Hessian match numerical differentiation."""
    field = CurvatureTaxisConfig().field()
    point = np.array([4.3, 4.8])
    gradient, hessian = field.derivatives(point)
    spacing = 1e-4
    axes = np.eye(2)
    numerical_gradient = np.array(
        [
            (
                field.concentration(point + spacing * axis)
                - field.concentration(point - spacing * axis)
            )
            / (2.0 * spacing)
            for axis in axes
        ],
    )
    numerical_hessian = np.empty((2, 2))
    center = field.concentration(point)
    for index, axis in enumerate(axes):
        numerical_hessian[index, index] = (
            field.concentration(point + spacing * axis)
            - 2.0 * center
            + field.concentration(point - spacing * axis)
        ) / spacing**2
    numerical_hessian[0, 1] = numerical_hessian[1, 0] = (
        field.concentration(point + spacing * axes[0] + spacing * axes[1])
        - field.concentration(point + spacing * axes[0] - spacing * axes[1])
        - field.concentration(point - spacing * axes[0] + spacing * axes[1])
        + field.concentration(point - spacing * axes[0] - spacing * axes[1])
    ) / (4.0 * spacing**2)

    assert gradient == pytest.approx(numerical_gradient, abs=2e-7)
    assert hessian == pytest.approx(numerical_hessian, abs=2e-6)


def test_observation_contains_nine_samples_and_no_reference_geometry():
    """The controller receives only the local stencil and its derived estimate."""
    observation = CurvatureTaxisEnvironment().reset()

    sample_names = {
        "center",
        "forward",
        "backward",
        "left",
        "right",
        "forward_left",
        "forward_right",
        "backward_left",
        "backward_right",
    }
    assert all(np.isfinite(getattr(observation.stencil, name)) for name in sample_names)
    assert not hasattr(observation, "reference_geometry")
    assert observation.geometry.gradient_magnitude > 0.0


@pytest.mark.parametrize("point", [(2.2, 2.6), (3.5, 5.1), (5.5, 4.7), (6.4, 6.0)])
def test_sensed_curvature_tracks_analytic_reference(point):
    """Nine local concentrations recover both geometric curvatures."""
    config = replace(CurvatureTaxisConfig(), start=point, initial_heading_degrees=67.0)
    environment = CurvatureTaxisEnvironment(config)
    sensed = environment.observe().geometry
    reference = environment.reference_geometry()

    assert sensed.streamline_curvature == pytest.approx(
        reference.streamline_curvature,
        abs=0.025,
    )
    assert sensed.level_set_curvature == pytest.approx(reference.level_set_curvature, abs=0.03)


def test_continuous_speed_law_is_bounded_and_decreasing():
    """Full-confidence speed falls continuously from its upper bound."""
    config = CurvatureTaxisConfig()
    values = np.array(
        [
            curvature_controlled_speed(
                curvature,
                1.0,
                config.min_speed,
                config.max_speed,
                config.curvature_scale,
                config.speed_exponent,
            )
            for curvature in np.linspace(0.0, 2.0, 100)
        ],
    )

    assert values[0] == pytest.approx(config.max_speed)
    assert np.all(np.diff(values) < 0.0)
    assert values[-1] > config.min_speed
    half = curvature_controlled_speed(
        config.curvature_scale,
        1.0,
        config.min_speed,
        config.max_speed,
        config.curvature_scale,
        config.speed_exponent,
    )
    assert half == pytest.approx((config.min_speed + config.max_speed) / 2.0)


def test_dynamics_use_reported_speed_and_turn_rate():
    """Independent speed and angular velocity drive the two state updates."""
    config = CurvatureTaxisConfig()
    environment = CurvatureTaxisEnvironment(config)
    old_position = environment.position
    old_heading = environment.heading

    _observation, reward, terminated, info = environment.step(0.7, 0.19)

    assert reward == 0.0
    assert not terminated
    assert environment.heading - old_heading == pytest.approx(0.7 * config.dt)
    assert np.linalg.norm(environment.position - old_position) == pytest.approx(0.19 * config.dt)
    assert info["speed"] == pytest.approx(0.19)
    assert info["turn_rate"] == pytest.approx(0.7)


def test_episode_starting_inside_target_terminates_without_moving():
    """An already satisfied initial state is not advanced by one artificial step."""
    config = replace(
        CurvatureTaxisConfig(),
        start=CurvatureTaxisConfig().source,
    )

    trace = run_taxis(config)

    assert trace.success
    assert trace.termination_reason is TerminationReason.TARGET_REACHED
    assert len(trace.snapshots) == 1
    assert not trace.transitions
    assert trace.snapshots[0].time == 0.0
    assert run_matched_constant_speed(config, trace).success


def test_controller_uses_sensed_geometry_and_bilateral_steering():
    """Locomotion reads field geometry while steering reads bilateral odor."""
    config = CurvatureTaxisConfig()
    observation = CurvatureTaxisEnvironment(config).observe()
    speed = adaptive_speed(observation, config)
    turn_rate = bilateral_turn_rate(
        observation,
        max_turn_rate=config.max_turn_rate,
        gain=config.steering_gain,
    )

    assert config.min_speed <= speed <= config.max_speed
    expected_sign = np.sign(observation.stencil.forward_left - observation.stencil.forward_right)
    assert np.sign(turn_rate) == expected_sign


def test_adaptive_trace_has_varying_curvature_speed_and_inverse_association():
    """The default run excites the intended curvature-speed relationship."""
    trace = run_taxis(CurvatureTaxisConfig())

    assert np.ptp(trace.estimated_streamline_curvatures) > 0.05
    assert np.ptp(trace.speeds) > 0.02
    assert np.corrcoef(np.abs(trace.estimated_streamline_curvatures), trace.speeds)[0, 1] < -0.5
    assert np.mean(
        trace.speeds[
            np.abs(trace.estimated_streamline_curvatures)
            >= np.quantile(np.abs(trace.estimated_streamline_curvatures), 0.75)
        ],
    ) < np.mean(
        trace.speeds[
            np.abs(trace.estimated_streamline_curvatures)
            <= np.quantile(np.abs(trace.estimated_streamline_curvatures), 0.25)
        ],
    )


def test_experienced_speeds_include_confidence_and_stay_below_full_confidence_envelope():
    """Plot points are exact policy outputs, not mislabeled envelope values."""
    config = CurvatureTaxisConfig()
    trace = run_taxis(config)
    for transition in trace.transitions:
        geometry = transition.observation.geometry
        expected = curvature_controlled_speed(
            geometry.streamline_curvature,
            geometry.confidence,
            config.min_speed,
            config.max_speed,
            config.curvature_scale,
            config.speed_exponent,
        )
        envelope = curvature_controlled_speed(
            geometry.streamline_curvature,
            1.0,
            config.min_speed,
            config.max_speed,
            config.curvature_scale,
            config.speed_exponent,
        )
        assert transition.command.speed == pytest.approx(expected)
        assert transition.command.speed <= envelope + 1e-12


def test_adaptive_navigation_reaches_source_and_matches_reference_well():
    """The locally sensed controller reaches the source with accurate geometry."""
    trace = run_taxis(CurvatureTaxisConfig())

    assert trace.success
    assert trace.termination_reason is TerminationReason.TARGET_REACHED
    assert trace.distances[-1] < CurvatureTaxisConfig().target_radius
    assert trace.concentrations[-1] > trace.concentrations[0]
    error = trace.estimated_streamline_curvatures - trace.reference_streamline_curvatures
    assert np.mean(np.abs(error)) < 0.03
    assert (
        np.corrcoef(trace.estimated_streamline_curvatures, trace.reference_streamline_curvatures)[
            0,
            1,
        ]
        > 0.95
    )


def test_matched_constant_baseline_has_same_mean_speed_and_steering_rule():
    """The comparison changes speed timing, not mean speed or steering logic."""
    config = CurvatureTaxisConfig()
    adaptive = run_taxis(config)
    constant = run_matched_constant_speed(config, adaptive)

    assert np.mean(constant.speeds) == pytest.approx(np.mean(adaptive.speeds))
    assert np.ptp(constant.speeds) == pytest.approx(0.0)
    first = constant.transitions[0]
    assert first.command.turn_rate == pytest.approx(
        bilateral_turn_rate(
            first.observation,
            max_turn_rate=config.max_turn_rate,
            gain=config.steering_gain,
        ),
    )


def test_heading_sweep_and_summary_are_complete():
    """Heading sweeps report both adaptive and matched policies."""
    rows = run_heading_sweep(CurvatureTaxisConfig(), count=4)
    summary = summarize_heading_sweep(rows)

    assert len(rows) == 4
    assert summary["heading_count"] == 4
    assert 0.0 <= summary["adaptive_success_rate"] <= 1.0
    assert 0.0 <= summary["constant_success_rate"] <= 1.0


def test_demo_artifacts_include_geometry_speed_and_aligned_next_samples(tmp_path):
    """Saved artifacts expose every measurement needed to audit the claim."""
    summary = save_demo_artifacts(tmp_path, CurvatureTaxisConfig(), heading_count=2)

    expected = {
        "curvature_taxis_demo.png",
        "adaptive_trace.csv",
        "matched_constant_trace.csv",
        "heading_sweep.csv",
        "summary.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert mpimg.imread(tmp_path / "curvature_taxis_demo.png").size > 0
    with (tmp_path / "adaptive_trace.csv").open(encoding="utf-8", newline="") as csv_file:
        first_row = next(csv.DictReader(csv_file))
    assert {
        "sample_center",
        "sample_forward_left",
        "sample_backward_right",
        "estimated_streamline_curvature",
        "estimated_level_set_curvature",
        "reference_streamline_curvature",
        "curvature_confidence",
        "speed",
        "turn_rate",
        "path_curvature",
        "next_sample_center",
        "next_sample_backward_right",
    } <= first_row.keys()
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["field"]["non_gaussian"]
    assert saved["field"]["non_circular"]
    assert saved["adaptive"]["success"]
    assert saved["estimator"]["correlation"] > 0.9
    assert saved["speed_relation"]["curvature_speed_correlation"] < 0.0
    assert summary["heading_sweep"]["heading_count"] == 2
