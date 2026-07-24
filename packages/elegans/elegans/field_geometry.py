"""Local geometric measurements of a two-dimensional scalar odor field.

The functions in this module deliberately distinguish geometry of the odor field from
geometry of an agent's path.  In particular, ``streamline_curvature`` describes how an
integral curve of the odor gradient bends in space; it is not an agent turn command.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

POSITION_DIMENSIONS = 2

type ConcentrationFunction = Callable[[NDArray[np.float64]], float]


@dataclass(frozen=True, slots=True)
class OdorStencil:
    """Nine concentration samples on a square stencil in the agent's body frame.

    ``forward`` and ``left`` define the positive body-frame axes.  Diagonal samples
    are displaced by ``spacing`` along both named axes, so they are at Euclidean
    distance ``sqrt(2) * spacing`` from the center.
    """

    center: float
    forward: float
    backward: float
    left: float
    right: float
    forward_left: float
    forward_right: float
    backward_left: float
    backward_right: float
    spacing: float

    def __post_init__(self) -> None:
        """Reject non-finite concentrations and invalid sensor spacing."""
        sample_names = (
            "center",
            "forward",
            "backward",
            "left",
            "right",
            "forward_left",
            "forward_right",
            "backward_left",
            "backward_right",
        )
        for name in sample_names:
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value):
                message = f"{name} must be a finite scalar"
                raise ValueError(message)
        _validate_positive_finite(self.spacing, "spacing")


@dataclass(frozen=True, slots=True)
class FieldGeometryEstimate:
    """Gradient, Hessian, and curvature estimates in ``(forward, left)`` axes.

    ``streamline_curvature`` is the signed curvature of a gradient-flow line: the
    spatial geometry an ideal gradient-following trajectory would encounter.  It is
    independent of, and must not be confused with, an agent's path curvature or turn
    action.  ``level_set_curvature`` is the signed curvature of an iso-concentration
    contour with the odor gradient as its normal.
    """

    gradient: tuple[float, float]
    hessian: tuple[tuple[float, float], tuple[float, float]]
    gradient_magnitude: float
    streamline_curvature: float
    level_set_curvature: float
    confidence: float


def sample_odor_stencil(
    concentration: ConcentrationFunction,
    position: Sequence[float] | NDArray[np.float64],
    heading: float,
    spacing: float,
) -> OdorStencil:
    """Sample a scalar odor field on a body-aligned nine-point stencil.

    Parameters
    ----------
    concentration
        Callable accepting a two-component world position and returning one scalar.
    position
        Center of the stencil in world ``(x, y)`` coordinates.
    heading
        Agent heading in radians.  Heading zero makes forward point along world +x.
    spacing
        Displacement along each body axis between adjacent stencil samples.
    """
    if not callable(concentration):
        message = "concentration must be callable"
        raise TypeError(message)

    center_position = np.asarray(position, dtype=np.float64)
    if center_position.shape != (POSITION_DIMENSIONS,) or not np.all(
        np.isfinite(center_position),
    ):
        message = "position must contain exactly two finite coordinates"
        raise ValueError(message)
    heading_value = _as_finite_scalar(heading, "heading")
    spacing_value = _validate_positive_finite(spacing, "spacing")

    forward_axis = np.array(
        [np.cos(heading_value), np.sin(heading_value)],
        dtype=np.float64,
    )
    left_axis = np.array([-forward_axis[1], forward_axis[0]], dtype=np.float64)
    forward_offset = spacing_value * forward_axis
    left_offset = spacing_value * left_axis

    def evaluate(offset: NDArray[np.float64]) -> float:
        return _as_finite_scalar(
            concentration(center_position + offset),
            "concentration",
        )

    zero_offset = np.zeros(POSITION_DIMENSIONS, dtype=np.float64)
    return OdorStencil(
        center=evaluate(zero_offset),
        forward=evaluate(forward_offset),
        backward=evaluate(-forward_offset),
        left=evaluate(left_offset),
        right=evaluate(-left_offset),
        forward_left=evaluate(forward_offset + left_offset),
        forward_right=evaluate(forward_offset - left_offset),
        backward_left=evaluate(-forward_offset + left_offset),
        backward_right=evaluate(-forward_offset - left_offset),
        spacing=spacing_value,
    )


def estimate_field_geometry(
    stencil: OdorStencil,
    gradient_floor: float,
) -> FieldGeometryEstimate:
    """Estimate local derivatives and field curvature with centered differences."""
    if not isinstance(stencil, OdorStencil):
        message = "stencil must be an OdorStencil"
        raise TypeError(message)
    gradient_floor_value = _validate_positive_finite(gradient_floor, "gradient_floor")

    spacing = stencil.spacing
    spacing_squared = spacing**2
    gradient_forward = (stencil.forward - stencil.backward) / (2.0 * spacing)
    gradient_left = (stencil.left - stencil.right) / (2.0 * spacing)
    hessian_forward_forward = (
        stencil.forward - 2.0 * stencil.center + stencil.backward
    ) / spacing_squared
    hessian_left_left = (stencil.left - 2.0 * stencil.center + stencil.right) / spacing_squared
    hessian_forward_left = (
        stencil.forward_left
        - stencil.forward_right
        - stencil.backward_left
        + stencil.backward_right
    ) / (4.0 * spacing_squared)

    return geometry_from_derivatives(
        (gradient_forward, gradient_left),
        (
            (hessian_forward_forward, hessian_forward_left),
            (hessian_forward_left, hessian_left_left),
        ),
        gradient_floor_value,
    )


def geometry_from_derivatives(
    gradient: Sequence[float] | NDArray[np.float64],
    hessian: Sequence[Sequence[float]] | NDArray[np.float64],
    gradient_floor: float,
) -> FieldGeometryEstimate:
    """Compute regularized field geometry from a gradient and Hessian.

    The input axes may be any right-handed orthonormal two-dimensional frame.  The
    returned derivative components retain that frame, while both curvature scalars
    are invariant to rotations of it.  The Hessian's symmetric part is used because
    only that part contributes to the curvature of a twice-differentiable field.

    ``gradient_floor`` regularizes curvature where the gradient direction is
    undefined.  The accompanying confidence is
    ``|gradient|^2 / (|gradient|^2 + gradient_floor^2)``.
    """
    gradient_array = np.asarray(gradient, dtype=np.float64)
    if gradient_array.shape != (POSITION_DIMENSIONS,) or not np.all(
        np.isfinite(gradient_array),
    ):
        message = "gradient must contain exactly two finite components"
        raise ValueError(message)

    hessian_array = np.asarray(hessian, dtype=np.float64)
    if hessian_array.shape != (POSITION_DIMENSIONS, POSITION_DIMENSIONS) or not np.all(
        np.isfinite(hessian_array),
    ):
        message = "hessian must have shape (2, 2) and contain only finite values"
        raise ValueError(message)
    gradient_floor_value = _validate_positive_finite(gradient_floor, "gradient_floor")

    symmetric_hessian = 0.5 * (hessian_array + hessian_array.T)
    gradient_forward, gradient_left = gradient_array
    hessian_forward_forward = symmetric_hessian[0, 0]
    hessian_forward_left = symmetric_hessian[0, 1]
    hessian_left_left = symmetric_hessian[1, 1]

    gradient_squared = float(gradient_array @ gradient_array)
    gradient_magnitude = float(np.sqrt(gradient_squared))
    regularized_squared = gradient_squared + gradient_floor_value**2
    curvature_denominator = regularized_squared**1.5

    level_set_numerator = (
        hessian_forward_forward * gradient_left**2
        - 2.0 * hessian_forward_left * gradient_forward * gradient_left
        + hessian_left_left * gradient_forward**2
    )
    streamline_numerator = (
        hessian_forward_left * (gradient_forward**2 - gradient_left**2)
        + (hessian_left_left - hessian_forward_forward) * gradient_forward * gradient_left
    )

    return FieldGeometryEstimate(
        gradient=(float(gradient_forward), float(gradient_left)),
        hessian=(
            (float(hessian_forward_forward), float(hessian_forward_left)),
            (float(hessian_forward_left), float(hessian_left_left)),
        ),
        gradient_magnitude=gradient_magnitude,
        streamline_curvature=float(streamline_numerator / curvature_denominator),
        level_set_curvature=float(level_set_numerator / curvature_denominator),
        confidence=float(gradient_squared / regularized_squared),
    )


def curvature_controlled_speed(  # noqa: PLR0913 - public scalar control-law API
    curvature: float,
    confidence: float,
    min_speed: float,
    max_speed: float,
    curvature_scale: float,
    exponent: float = 2.0,
) -> float:
    """Map sensed field curvature to a smooth bounded forward speed.

    With full confidence, zero curvature produces ``max_speed`` and increasing
    curvature magnitude continuously approaches ``min_speed``.  Low confidence
    conservatively blends the result toward ``min_speed``.
    """
    curvature_value = _as_finite_scalar(curvature, "curvature")
    confidence_value = _as_finite_scalar(confidence, "confidence")
    min_speed_value = _as_finite_scalar(min_speed, "min_speed")
    max_speed_value = _as_finite_scalar(max_speed, "max_speed")
    curvature_scale_value = _validate_positive_finite(curvature_scale, "curvature_scale")
    exponent_value = _validate_positive_finite(exponent, "exponent")

    if not 0.0 <= confidence_value <= 1.0:
        message = "confidence must lie in [0, 1]"
        raise ValueError(message)
    if min_speed_value < 0.0:
        message = "min_speed must be greater than or equal to zero"
        raise ValueError(message)
    if max_speed_value <= 0.0 or max_speed_value < min_speed_value:
        message = "max_speed must be positive and greater than or equal to min_speed"
        raise ValueError(message)

    normalized_curvature = abs(curvature_value) / curvature_scale_value
    response = confidence_value / (1.0 + normalized_curvature**exponent_value)
    return float(min_speed_value + (max_speed_value - min_speed_value) * response)


def _validate_positive_finite(value: object, name: str) -> float:
    """Require one scalar to be finite and strictly positive, then return it."""
    numeric_value = _as_finite_scalar(value, name)
    if numeric_value <= 0.0:
        message = f"{name} must be a finite scalar greater than zero"
        raise ValueError(message)
    return numeric_value


def _as_finite_scalar(value: object, name: str) -> float:
    """Convert a scalar-like value to float while rejecting arrays and non-finite values."""
    try:
        value_array = np.asarray(value)
        numeric_value = float(value_array.item())
    except (TypeError, ValueError):
        message = f"{name} must be a finite scalar"
        raise ValueError(message) from None
    if value_array.shape != () or not np.isfinite(numeric_value):
        message = f"{name} must be a finite scalar"
        raise ValueError(message)
    return numeric_value
