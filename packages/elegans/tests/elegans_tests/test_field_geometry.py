"""Tests for local scalar-field geometry sensing and speed modulation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from elegans.field_geometry import (
    FieldGeometryEstimate,
    OdorStencil,
    curvature_controlled_speed,
    estimate_field_geometry,
    geometry_from_derivatives,
    sample_odor_stencil,
)


def test_quadratic_stencil_recovers_body_frame_gradient_and_hessian_exactly():
    """Centered differences exactly recover every derivative of a quadratic field."""
    world_gradient_at_origin = np.array([0.7, -1.1])
    world_hessian = np.array([[1.4, -0.35], [-0.35, 0.6]])
    position = np.array([0.8, -0.4])
    heading = 0.73
    spacing = 0.08
    gradient_floor = 1e-7

    def quadratic(point: np.ndarray) -> float:
        return float(2.3 + world_gradient_at_origin @ point + 0.5 * point @ world_hessian @ point)

    forward = np.array([np.cos(heading), np.sin(heading)])
    left = np.array([-np.sin(heading), np.cos(heading)])
    body_to_world = np.column_stack((forward, left))
    expected_world_gradient = world_gradient_at_origin + world_hessian @ position
    expected_body_gradient = body_to_world.T @ expected_world_gradient
    expected_body_hessian = body_to_world.T @ world_hessian @ body_to_world

    stencil = sample_odor_stencil(quadratic, position, heading, spacing)
    estimate = estimate_field_geometry(stencil, gradient_floor)
    reference = geometry_from_derivatives(
        expected_body_gradient,
        expected_body_hessian,
        gradient_floor,
    )

    assert estimate.gradient == pytest.approx(expected_body_gradient, abs=1e-12)
    assert np.asarray(estimate.hessian) == pytest.approx(expected_body_hessian, abs=1e-11)
    assert estimate.gradient_magnitude == pytest.approx(reference.gradient_magnitude)
    assert estimate.streamline_curvature == pytest.approx(reference.streamline_curvature)
    assert estimate.level_set_curvature == pytest.approx(reference.level_set_curvature)
    assert estimate.confidence == pytest.approx(reference.confidence)


def test_radial_field_has_curved_level_sets_but_straight_gradient_flow():
    """Concentric contours curve even though radial gradient streamlines do not."""
    radius = 2.5
    gradient_floor = 1e-9
    estimate = geometry_from_derivatives(
        (radius, 0.0),
        ((1.0, 0.0), (0.0, 1.0)),
        gradient_floor,
    )

    assert estimate.streamline_curvature == pytest.approx(0.0, abs=1e-15)
    assert estimate.level_set_curvature == pytest.approx(1.0 / radius)


def test_straight_parallel_level_sets_have_zero_curvature():
    """A linear field has neither level-set nor gradient-flow bending."""
    estimate = geometry_from_derivatives(
        (3.0, -4.0),
        ((0.0, 0.0), (0.0, 0.0)),
        1e-6,
    )

    assert estimate.gradient_magnitude == pytest.approx(5.0)
    assert estimate.streamline_curvature == pytest.approx(0.0)
    assert estimate.level_set_curvature == pytest.approx(0.0)


def test_curvatures_are_invariant_to_rotation_of_derivative_axes():
    """Changing orthonormal coordinates preserves both geometric curvatures."""
    gradient = np.array([1.3, -0.8])
    hessian = np.array([[0.7, 0.4], [0.4, -0.2]])
    angle = 1.17
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
    )
    gradient_floor = 0.03

    original = geometry_from_derivatives(gradient, hessian, gradient_floor)
    rotated = geometry_from_derivatives(
        rotation.T @ gradient,
        rotation.T @ hessian @ rotation,
        gradient_floor,
    )

    assert rotated.gradient_magnitude == pytest.approx(original.gradient_magnitude)
    assert rotated.streamline_curvature == pytest.approx(original.streamline_curvature)
    assert rotated.level_set_curvature == pytest.approx(original.level_set_curvature)
    assert rotated.confidence == pytest.approx(original.confidence)


def test_geometry_is_invariant_to_positive_rescaling_with_scaled_floor():
    """Changing odor units does not alter geometry when its noise floor scales too."""
    gradient = np.array([0.4, 0.9])
    hessian = np.array([[1.2, -0.3], [-0.3, 0.5]])
    gradient_floor = 0.02
    scale = 7.5

    original = geometry_from_derivatives(gradient, hessian, gradient_floor)
    rescaled = geometry_from_derivatives(
        scale * gradient,
        scale * hessian,
        scale * gradient_floor,
    )

    assert rescaled.streamline_curvature == pytest.approx(original.streamline_curvature)
    assert rescaled.level_set_curvature == pytest.approx(original.level_set_curvature)
    assert rescaled.confidence == pytest.approx(original.confidence)


def test_zero_gradient_is_finite_and_has_zero_confidence():
    """Critical points produce a cautious finite estimate instead of division by zero."""
    estimate = geometry_from_derivatives(
        (0.0, 0.0),
        ((10.0, -4.0), (-4.0, -7.0)),
        0.05,
    )

    assert estimate.gradient_magnitude == 0.0
    assert estimate.confidence == 0.0
    assert estimate.streamline_curvature == 0.0
    assert estimate.level_set_curvature == 0.0
    assert np.all(np.isfinite(np.asarray(estimate.gradient)))
    assert np.all(np.isfinite(np.asarray(estimate.hessian)))


def test_weak_gradient_confidence_matches_continuous_formula():
    """Confidence continuously represents gradient strength relative to the floor."""
    gradient_floor = 0.2
    estimate = geometry_from_derivatives(
        (0.12, 0.16),
        ((1.0, 0.2), (0.2, 0.5)),
        gradient_floor,
    )

    gradient_squared = 0.12**2 + 0.16**2
    expected = gradient_squared / (gradient_squared + gradient_floor**2)
    assert estimate.confidence == pytest.approx(expected)
    assert np.isfinite(estimate.streamline_curvature)
    assert np.isfinite(estimate.level_set_curvature)


def test_speed_is_bounded_symmetric_and_decreases_with_curvature():
    """The continuous motor law slows monotonically for either curvature sign."""
    magnitudes = np.linspace(0.0, 8.0, 81)
    speeds = np.array(
        [
            curvature_controlled_speed(
                curvature,
                confidence=1.0,
                min_speed=0.1,
                max_speed=0.9,
                curvature_scale=1.5,
            )
            for curvature in magnitudes
        ],
    )

    assert speeds[0] == pytest.approx(0.9)
    assert np.all(np.diff(speeds) < 0.0)
    assert np.all((speeds >= 0.1) & (speeds <= 0.9))
    assert curvature_controlled_speed(-2.0, 1.0, 0.1, 0.9, 1.5) == pytest.approx(
        curvature_controlled_speed(2.0, 1.0, 0.1, 0.9, 1.5),
    )


def test_low_confidence_blends_speed_toward_safe_minimum():
    """An uncertain curvature estimate cannot be mistaken for fast, flat terrain."""
    assert curvature_controlled_speed(0.0, 0.0, 0.15, 0.8, 1.0) == pytest.approx(0.15)
    assert curvature_controlled_speed(0.0, 0.5, 0.15, 0.8, 1.0) == pytest.approx(0.475)


def test_public_records_are_immutable():
    """Sensory inputs and geometry results are stable value objects."""
    stencil = OdorStencil(*(1.0 for _ in range(9)), spacing=0.1)
    estimate = estimate_field_geometry(stencil, gradient_floor=0.01)

    with pytest.raises(FrozenInstanceError):
        stencil.center = 2.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        estimate.confidence = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("gradient", "hessian", "gradient_floor"),
    [
        ((1.0,), ((1.0, 0.0), (0.0, 1.0)), 0.1),
        ((1.0, np.nan), ((1.0, 0.0), (0.0, 1.0)), 0.1),
        ((1.0, 0.0), ((1.0,), (0.0,)), 0.1),
        ((1.0, 0.0), ((1.0, np.inf), (0.0, 1.0)), 0.1),
        ((1.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), 0.0),
    ],
)
def test_derivative_validation_rejects_bad_shapes_and_values(
    gradient,
    hessian,
    gradient_floor,
):
    """Malformed derivative inputs fail before producing misleading geometry."""
    with pytest.raises(ValueError, match="must"):
        geometry_from_derivatives(gradient, hessian, gradient_floor)


@pytest.mark.parametrize(
    ("curvature", "confidence", "min_speed", "max_speed", "scale", "exponent"),
    [
        (np.nan, 1.0, 0.1, 1.0, 1.0, 2.0),
        (0.0, -0.1, 0.1, 1.0, 1.0, 2.0),
        (0.0, 1.1, 0.1, 1.0, 1.0, 2.0),
        (0.0, 1.0, -0.1, 1.0, 1.0, 2.0),
        (0.0, 1.0, 0.5, 0.4, 1.0, 2.0),
        (0.0, 1.0, 0.1, 1.0, 0.0, 2.0),
        (0.0, 1.0, 0.1, 1.0, 1.0, 0.0),
    ],
)
def test_speed_validation_rejects_invalid_parameters(  # noqa: PLR0913
    curvature,
    confidence,
    min_speed,
    max_speed,
    scale,
    exponent,
):
    """The speed law rejects values outside its physical domain."""
    with pytest.raises(ValueError, match="must"):
        curvature_controlled_speed(
            curvature,
            confidence,
            min_speed,
            max_speed,
            scale,
            exponent,
        )


def test_stencil_validation_rejects_invalid_geometry_and_samples():
    """Stencil sampling validates its geometry and every field response."""
    with pytest.raises(ValueError, match="position"):
        sample_odor_stencil(lambda point: float(point[0]), (0.0,), 0.0, 0.1)
    with pytest.raises(ValueError, match="heading"):
        sample_odor_stencil(lambda point: float(point[0]), (0.0, 0.0), np.inf, 0.1)
    with pytest.raises(ValueError, match="spacing"):
        sample_odor_stencil(lambda point: float(point[0]), (0.0, 0.0), 0.0, -0.1)
    with pytest.raises(ValueError, match="concentration"):
        sample_odor_stencil(lambda _point: np.nan, (0.0, 0.0), 0.0, 0.1)


def test_hessian_is_symmetrized_before_geometry_is_computed():
    """Only the Hessian's geometrically meaningful symmetric part is retained."""
    estimate = geometry_from_derivatives(
        (1.0, 2.0),
        ((3.0, 5.0), (1.0, 4.0)),
        0.1,
    )

    assert estimate.hessian == ((3.0, 3.0), (3.0, 4.0))
    assert isinstance(estimate, FieldGeometryEstimate)
