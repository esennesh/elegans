"""Curvature-sensing taxis in a smooth, non-Gaussian odor field.

The agent receives nine local concentration samples.  It estimates the gradient,
Hessian, and curvature of the odor-gradient streamlines from those samples, then
uses field curvature to set its forward speed.  Steering remains a separate
bilateral reflex: it is a turn rate, not the curvature measurement.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from elegans.field_geometry import (
    FieldGeometryEstimate,
    OdorStencil,
    curvature_controlled_speed,
    estimate_field_geometry,
    geometry_from_derivatives,
    sample_odor_stencil,
)

if TYPE_CHECKING:
    from pathlib import Path

POSITION_DIMENSIONS = 2
MINIMUM_CORRELATION_SAMPLES = 2
CONSTANT_TOLERANCE = 1e-15
DEFAULT_VIDEO_FPS = 25
DEFAULT_VIDEO_PLAYBACK_SPEED = 2.0
DEFAULT_VIDEO_DPI = 100
DEFAULT_VIDEO_BITRATE_KBPS = 280
VIDEO_END_HOLD_SECONDS = 0.8
VIDEO_FIELD_GRID_SIZE = 180
VIDEO_STREAM_GRID_SIZE = 23
MINIMUM_TRACE_SNAPSHOTS = 2
STENCIL_NAMES = (
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

type ConcentrationFunction = Callable[[NDArray[np.float64]], float]


class TerminationReason(StrEnum):
    """Reasons an episode can end."""

    TARGET_REACHED = "target_reached"
    ARENA_EXITED = "arena_exited"
    TIME_LIMIT = "time_limit"


@dataclass(frozen=True, slots=True)
class CurvedOdorField:
    """A rotated anisotropic quartic field with one unique maximum.

    In dimensionless rotated coordinates ``U, V``, concentration is

    ``exp(-0.5 * (U**2 + V**2) - beta * U**4 - gamma * U**2 * V**2)``.

    Positive ``beta`` and ``gamma`` make this genuinely non-Gaussian.  Unequal
    scales and rotation make the field non-circular, while the quartic coupling
    bends its gradient-flow streamlines.
    """

    source: tuple[float, float]
    scale_u: float
    scale_v: float
    rotation_degrees: float
    beta: float
    gamma: float

    def __post_init__(self) -> None:
        """Reject invalid field parameters even when constructed directly."""
        _point(self.source)
        positive = {
            "scale_u": self.scale_u,
            "scale_v": self.scale_v,
            "beta": self.beta,
            "gamma": self.gamma,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                message = f"{name} must be finite and greater than zero"
                raise ValueError(message)
        if not np.isfinite(self.rotation_degrees):
            message = "rotation_degrees must be finite"
            raise ValueError(message)

    def _transform(self) -> NDArray[np.float64]:
        angle = np.deg2rad(self.rotation_degrees)
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        return np.array(
            [
                [cosine / self.scale_u, sine / self.scale_u],
                [-sine / self.scale_v, cosine / self.scale_v],
            ],
            dtype=np.float64,
        )

    def concentration(self, position: Sequence[float] | NDArray[np.float64]) -> float:
        """Evaluate concentration at a 2-D position."""
        point = _point(position)
        coordinates = self._transform() @ (point - np.asarray(self.source, dtype=np.float64))
        u_coordinate, v_coordinate = coordinates
        energy = (
            0.5 * (u_coordinate**2 + v_coordinate**2)
            + self.beta * u_coordinate**4
            + self.gamma * u_coordinate**2 * v_coordinate**2
        )
        return float(np.exp(-energy))

    def derivatives(
        self,
        position: Sequence[float] | NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the analytic world-coordinate gradient and Hessian."""
        point = _point(position)
        transform = self._transform()
        u_coordinate, v_coordinate = transform @ (point - np.asarray(self.source, dtype=np.float64))
        energy = (
            0.5 * (u_coordinate**2 + v_coordinate**2)
            + self.beta * u_coordinate**4
            + self.gamma * u_coordinate**2 * v_coordinate**2
        )
        concentration = float(np.exp(-energy))
        energy_gradient_local = np.array(
            [
                u_coordinate
                + 4.0 * self.beta * u_coordinate**3
                + 2.0 * self.gamma * u_coordinate * v_coordinate**2,
                v_coordinate + 2.0 * self.gamma * u_coordinate**2 * v_coordinate,
            ],
            dtype=np.float64,
        )
        energy_hessian_local = np.array(
            [
                [
                    1.0 + 12.0 * self.beta * u_coordinate**2 + 2.0 * self.gamma * v_coordinate**2,
                    4.0 * self.gamma * u_coordinate * v_coordinate,
                ],
                [
                    4.0 * self.gamma * u_coordinate * v_coordinate,
                    1.0 + 2.0 * self.gamma * u_coordinate**2,
                ],
            ],
            dtype=np.float64,
        )
        energy_gradient = transform.T @ energy_gradient_local
        energy_hessian = transform.T @ energy_hessian_local @ transform
        gradient = -concentration * energy_gradient
        hessian = concentration * (np.outer(energy_gradient, energy_gradient) - energy_hessian)
        return gradient, hessian


@dataclass(frozen=True, slots=True)
class CurvatureTaxisConfig:
    """Parameters for the curvature-sensing navigation experiment."""

    arena_size: float = 10.0
    source: tuple[float, float] = (7.7, 6.4)
    start: tuple[float, float] = (1.7, 2.1)
    initial_heading_degrees: float = 28.0
    field_scale_u: float = 3.8
    field_scale_v: float = 1.45
    field_rotation_degrees: float = 31.0
    field_beta: float = 0.035
    field_gamma: float = 0.24
    sensor_spacing: float = 0.16
    gradient_floor: float = 2e-4
    min_speed: float = 0.10
    max_speed: float = 0.42
    curvature_scale: float = 0.22
    speed_exponent: float = 2.0
    max_turn_rate: float = 1.25
    steering_gain: float = 2.4
    dt: float = 0.04
    target_radius: float = 0.45
    max_duration: float = 120.0

    def __post_init__(self) -> None:
        """Reject invalid geometry and dynamics."""
        positive = {
            "arena_size": self.arena_size,
            "field_scale_u": self.field_scale_u,
            "field_scale_v": self.field_scale_v,
            "sensor_spacing": self.sensor_spacing,
            "gradient_floor": self.gradient_floor,
            "min_speed": self.min_speed,
            "max_speed": self.max_speed,
            "curvature_scale": self.curvature_scale,
            "speed_exponent": self.speed_exponent,
            "max_turn_rate": self.max_turn_rate,
            "steering_gain": self.steering_gain,
            "dt": self.dt,
            "target_radius": self.target_radius,
            "max_duration": self.max_duration,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                message = f"{name} must be finite and greater than zero"
                raise ValueError(message)
        if self.min_speed > self.max_speed:
            message = "min_speed must not exceed max_speed"
            raise ValueError(message)
        if (
            not np.isfinite(self.field_beta)
            or not np.isfinite(self.field_gamma)
            or self.field_beta <= 0.0
            or self.field_gamma <= 0.0
        ):
            message = "field_beta and field_gamma must be finite and greater than zero"
            raise ValueError(message)
        if not np.isfinite(self.field_rotation_degrees):
            message = "field_rotation_degrees must be finite"
            raise ValueError(message)
        if not np.isfinite(self.initial_heading_degrees):
            message = "initial_heading_degrees must be finite"
            raise ValueError(message)
        for name, point in (("source", self.source), ("start", self.start)):
            checked = _point(point)
            if not np.all((checked >= 0.0) & (checked <= self.arena_size)):
                message = f"{name} must lie inside the arena"
                raise ValueError(message)

    def field(self) -> CurvedOdorField:
        """Construct the configured odor field."""
        return CurvedOdorField(
            source=self.source,
            scale_u=self.field_scale_u,
            scale_v=self.field_scale_v,
            rotation_degrees=self.field_rotation_degrees,
            beta=self.field_beta,
            gamma=self.field_gamma,
        )


@dataclass(frozen=True, slots=True)
class TaxisObservation:
    """Raw local odor samples and the agent's estimate derived from them."""

    stencil: OdorStencil
    geometry: FieldGeometryEstimate


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """Separate steering and locomotion outputs."""

    turn_rate: float
    speed: float


@dataclass(frozen=True, slots=True)
class TaxisSnapshot:
    """State, sensing, and analytic reference at one instant."""

    time: float
    position: tuple[float, float]
    heading: float
    observation: TaxisObservation
    reference_geometry: FieldGeometryEstimate
    distance_to_source: float

    @property
    def center_concentration(self) -> float:
        """Return concentration at the agent center."""
        return self.observation.stencil.center


@dataclass(frozen=True, slots=True)
class TaxisTransition:
    """An aligned sensed-state, action, next-sensed-state transition."""

    observation: TaxisObservation
    command: ControlCommand
    next_observation: TaxisObservation


@dataclass(frozen=True, slots=True)
class TaxisTrace:
    """A complete episode."""

    snapshots: tuple[TaxisSnapshot, ...]
    transitions: tuple[TaxisTransition, ...]
    success: bool
    termination_reason: TerminationReason
    speed_policy: str

    @property
    def times(self) -> NDArray[np.float64]:
        """Return snapshot times."""
        return np.fromiter((item.time for item in self.snapshots), dtype=np.float64)

    @property
    def positions(self) -> NDArray[np.float64]:
        """Return the two-dimensional trajectory."""
        return np.asarray([item.position for item in self.snapshots], dtype=np.float64)

    @property
    def distances(self) -> NDArray[np.float64]:
        """Return distances to the field maximum."""
        return np.fromiter((item.distance_to_source for item in self.snapshots), dtype=np.float64)

    @property
    def concentrations(self) -> NDArray[np.float64]:
        """Return center concentrations."""
        return np.fromiter((item.center_concentration for item in self.snapshots), dtype=np.float64)

    @property
    def speeds(self) -> NDArray[np.float64]:
        """Return applied forward speeds."""
        return np.fromiter((item.command.speed for item in self.transitions), dtype=np.float64)

    @property
    def turn_rates(self) -> NDArray[np.float64]:
        """Return applied angular velocities."""
        return np.fromiter((item.command.turn_rate for item in self.transitions), dtype=np.float64)

    @property
    def path_curvatures(self) -> NDArray[np.float64]:
        """Return realized path curvature, angular velocity divided by speed."""
        return self.turn_rates / self.speeds

    @property
    def estimated_streamline_curvatures(self) -> NDArray[np.float64]:
        """Return locally estimated odor-streamline curvature."""
        return np.fromiter(
            (item.observation.geometry.streamline_curvature for item in self.transitions),
            dtype=np.float64,
        )

    @property
    def reference_streamline_curvatures(self) -> NDArray[np.float64]:
        """Return analytic odor-streamline curvature."""
        return np.fromiter(
            (item.reference_geometry.streamline_curvature for item in self.snapshots[:-1]),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class HeadingSweepRow:
    """Adaptive and matched-constant outcomes for one initial heading."""

    heading_degrees: float
    adaptive_success: bool
    adaptive_time: float
    adaptive_final_distance: float
    adaptive_mean_speed: float
    constant_success: bool
    constant_time: float
    constant_final_distance: float


class HeadingSweepSummary(TypedDict):
    """Aggregate heading-sweep metrics."""

    heading_count: int
    adaptive_successes: int
    constant_successes: int
    adaptive_success_rate: float
    constant_success_rate: float
    median_adaptive_time_to_target: float | None
    median_constant_time_to_target: float | None
    median_adaptive_final_distance: float
    median_constant_final_distance: float


class CurvatureTaxisEnvironment:
    """Continuous point-agent environment with local curvature sensing."""

    def __init__(self, config: CurvatureTaxisConfig | None = None) -> None:
        self.config = config or CurvatureTaxisConfig()
        self.field = self.config.field()
        self._source = np.asarray(self.config.source, dtype=np.float64)
        self._position = np.empty(POSITION_DIMENSIONS, dtype=np.float64)
        self._heading = 0.0
        self._time = 0.0
        self._terminated = False
        self._termination_reason: TerminationReason | None = None
        self.reset()

    @property
    def position(self) -> NDArray[np.float64]:
        """Return a copy of the agent position."""
        return self._position.copy()

    @property
    def heading(self) -> float:
        """Return heading in radians."""
        return self._heading

    @property
    def time(self) -> float:
        """Return elapsed simulated time."""
        return self._time

    @property
    def terminated(self) -> bool:
        """Return whether the current episode has ended."""
        return self._terminated

    @property
    def termination_reason(self) -> TerminationReason | None:
        """Return the reason for termination, if the episode has ended."""
        return self._termination_reason

    def concentration(self, position: Sequence[float] | NDArray[np.float64]) -> float:
        """Evaluate the configured field."""
        return self.field.concentration(position)

    def reset(self) -> TaxisObservation:
        """Restore the configured initial state and observe it."""
        self._position = np.asarray(self.config.start, dtype=np.float64)
        self._heading = float(np.deg2rad(self.config.initial_heading_degrees))
        self._time = 0.0
        initial_distance = float(np.linalg.norm(self._position - self._source))
        self._terminated = initial_distance < self.config.target_radius
        self._termination_reason = TerminationReason.TARGET_REACHED if self._terminated else None
        return self.observe()

    def observe(self) -> TaxisObservation:
        """Sense nine concentrations and derive geometry from only those samples."""
        stencil = sample_odor_stencil(
            self.concentration,
            self._position,
            self._heading,
            self.config.sensor_spacing,
        )
        return TaxisObservation(
            stencil=stencil,
            geometry=estimate_field_geometry(stencil, self.config.gradient_floor),
        )

    def reference_geometry(self) -> FieldGeometryEstimate:
        """Return an analytic reference in the agent's forward/left frame."""
        gradient_world, hessian_world = self.field.derivatives(self._position)
        forward = np.array([np.cos(self._heading), np.sin(self._heading)], dtype=np.float64)
        left = np.array([-forward[1], forward[0]], dtype=np.float64)
        basis = np.column_stack((forward, left))
        gradient_local = basis.T @ gradient_world
        hessian_local = basis.T @ hessian_world @ basis
        return geometry_from_derivatives(
            gradient_local,
            hessian_local,
            self.config.gradient_floor,
        )

    def snapshot(self) -> TaxisSnapshot:
        """Capture state, sensed geometry, and analytic reference."""
        return TaxisSnapshot(
            time=self._time,
            position=(float(self._position[0]), float(self._position[1])),
            heading=self._heading,
            observation=self.observe(),
            reference_geometry=self.reference_geometry(),
            distance_to_source=float(np.linalg.norm(self._position - self._source)),
        )

    def step(
        self,
        turn_rate: float,
        speed: float,
    ) -> tuple[TaxisObservation, float, bool, dict[str, float | bool | str]]:
        """Advance using independent angular velocity and forward speed."""
        if self._terminated:
            message = "step() called after termination; call reset() first"
            raise RuntimeError(message)
        if not np.isfinite(turn_rate) or not np.isfinite(speed):
            message = "turn_rate and speed must be finite"
            raise ValueError(message)
        applied_turn_rate = float(
            np.clip(turn_rate, -self.config.max_turn_rate, self.config.max_turn_rate),
        )
        applied_speed = float(np.clip(speed, self.config.min_speed, self.config.max_speed))

        self._heading += applied_turn_rate * self.config.dt
        direction = np.array([np.cos(self._heading), np.sin(self._heading)], dtype=np.float64)
        self._position += applied_speed * direction * self.config.dt
        self._time += self.config.dt

        snapshot = self.snapshot()
        if snapshot.distance_to_source < self.config.target_radius:
            self._termination_reason = TerminationReason.TARGET_REACHED
        elif not self._inside_arena():
            self._termination_reason = TerminationReason.ARENA_EXITED
        elif self._time >= self.config.max_duration:
            self._termination_reason = TerminationReason.TIME_LIMIT
        self._terminated = self._termination_reason is not None
        return (
            snapshot.observation,
            0.0,
            self._terminated,
            {
                "time": snapshot.time,
                "turn_rate": applied_turn_rate,
                "speed": applied_speed,
                "distance_to_source": snapshot.distance_to_source,
                "success": self._termination_reason is TerminationReason.TARGET_REACHED,
                "termination_reason": self._termination_reason.value
                if self._termination_reason
                else "",
            },
        )

    def _inside_arena(self) -> bool:
        return bool(np.all((self._position >= 0.0) & (self._position <= self.config.arena_size)))


def bilateral_turn_rate(
    observation: TaxisObservation,
    *,
    max_turn_rate: float,
    gain: float,
) -> float:
    """Turn toward the stronger forward-side sample."""
    left = observation.stencil.forward_left
    right = observation.stencil.forward_right
    error = (left - right) / (left + right + 1e-12)
    return float(max_turn_rate * np.tanh(gain * error))


def adaptive_speed(observation: TaxisObservation, config: CurvatureTaxisConfig) -> float:
    """Map sensed field curvature and confidence to forward speed."""
    geometry = observation.geometry
    return curvature_controlled_speed(
        geometry.streamline_curvature,
        geometry.confidence,
        config.min_speed,
        config.max_speed,
        config.curvature_scale,
        config.speed_exponent,
    )


def run_episode(
    config: CurvatureTaxisConfig,
    *,
    constant_speed: float | None = None,
    steering: bool = True,
) -> TaxisTrace:
    """Run one episode with adaptive or fixed speed and the same steering reflex."""
    environment = CurvatureTaxisEnvironment(config)
    observation = environment.reset()
    snapshots = [environment.snapshot()]
    transitions: list[TaxisTransition] = []
    if environment.terminated:
        reason = environment.termination_reason
        if reason is None:  # pragma: no cover - guarded by environment invariant
            message = "terminated environment must have a termination reason"
            raise RuntimeError(message)
        return TaxisTrace(
            snapshots=tuple(snapshots),
            transitions=(),
            success=reason is TerminationReason.TARGET_REACHED,
            termination_reason=reason,
            speed_policy="adaptive" if constant_speed is None else "constant",
        )
    while True:
        turn_rate = (
            bilateral_turn_rate(
                observation,
                max_turn_rate=config.max_turn_rate,
                gain=config.steering_gain,
            )
            if steering
            else 0.0
        )
        requested_speed = (
            adaptive_speed(observation, config) if constant_speed is None else constant_speed
        )
        next_observation, _reward, terminated, info = environment.step(turn_rate, requested_speed)
        command = ControlCommand(
            turn_rate=float(info["turn_rate"]),
            speed=float(info["speed"]),
        )
        transitions.append(
            TaxisTransition(
                observation=observation,
                command=command,
                next_observation=next_observation,
            ),
        )
        snapshots.append(environment.snapshot())
        observation = next_observation
        if terminated:
            reason = TerminationReason(str(info["termination_reason"]))
            return TaxisTrace(
                snapshots=tuple(snapshots),
                transitions=tuple(transitions),
                success=reason is TerminationReason.TARGET_REACHED,
                termination_reason=reason,
                speed_policy="adaptive" if constant_speed is None else "constant",
            )


def run_taxis(config: CurvatureTaxisConfig) -> TaxisTrace:
    """Run curvature-aware adaptive-speed taxis."""
    return run_episode(config)


def run_matched_constant_speed(
    config: CurvatureTaxisConfig,
    adaptive_trace: TaxisTrace | None = None,
) -> TaxisTrace:
    """Run identical steering at the adaptive trace's mean speed."""
    reference = adaptive_trace or run_taxis(config)
    matched_speed = float(np.mean(reference.speeds)) if reference.transitions else config.min_speed
    return run_episode(config, constant_speed=matched_speed)


def run_no_steering(config: CurvatureTaxisConfig) -> TaxisTrace:
    """Run a no-steering control with fixed maximum speed."""
    return run_episode(config, constant_speed=config.max_speed, steering=False)


def run_heading_sweep(
    config: CurvatureTaxisConfig,
    count: int = 20,
) -> tuple[HeadingSweepRow, ...]:
    """Compare adaptive and matched-constant speed across initial headings."""
    if count < 1:
        message = "count must be at least one"
        raise ValueError(message)
    rows: list[HeadingSweepRow] = []
    for heading in np.linspace(0.0, 360.0, count, endpoint=False):
        heading_config = replace(config, initial_heading_degrees=float(heading))
        adaptive = run_taxis(heading_config)
        constant = run_matched_constant_speed(heading_config, adaptive)
        rows.append(
            HeadingSweepRow(
                heading_degrees=float(heading),
                adaptive_success=adaptive.success,
                adaptive_time=adaptive.snapshots[-1].time,
                adaptive_final_distance=adaptive.distances[-1],
                adaptive_mean_speed=float(np.mean(adaptive.speeds)),
                constant_success=constant.success,
                constant_time=constant.snapshots[-1].time,
                constant_final_distance=constant.distances[-1],
            ),
        )
    return tuple(rows)


def summarize_heading_sweep(rows: Sequence[HeadingSweepRow]) -> HeadingSweepSummary:
    """Summarize robustness across initial headings."""
    if not rows:
        message = "rows must not be empty"
        raise ValueError(message)
    successful_adaptive_times = [row.adaptive_time for row in rows if row.adaptive_success]
    successful_constant_times = [row.constant_time for row in rows if row.constant_success]
    return {
        "heading_count": len(rows),
        "adaptive_successes": sum(row.adaptive_success for row in rows),
        "constant_successes": sum(row.constant_success for row in rows),
        "adaptive_success_rate": float(np.mean([row.adaptive_success for row in rows])),
        "constant_success_rate": float(np.mean([row.constant_success for row in rows])),
        "median_adaptive_time_to_target": (
            float(np.median(successful_adaptive_times)) if successful_adaptive_times else None
        ),
        "median_constant_time_to_target": (
            float(np.median(successful_constant_times)) if successful_constant_times else None
        ),
        "median_adaptive_final_distance": float(
            np.median([row.adaptive_final_distance for row in rows]),
        ),
        "median_constant_final_distance": float(
            np.median([row.constant_final_distance for row in rows]),
        ),
    }


def save_demo_artifacts(  # noqa: PLR0913 - optional video settings share artifact API
    output_dir: Path,
    config: CurvatureTaxisConfig,
    *,
    heading_count: int = 20,
    render_video: bool = False,
    video_fps: int = DEFAULT_VIDEO_FPS,
    video_playback_speed: float = DEFAULT_VIDEO_PLAYBACK_SPEED,
) -> dict[str, Any]:
    """Run the experiment and save plots, traces, metrics, and optional video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    adaptive = run_taxis(config)
    constant = run_matched_constant_speed(config, adaptive)
    no_steering = run_no_steering(config)
    sweep = run_heading_sweep(config, heading_count)

    figure_path = output_dir / "curvature_taxis_demo.png"
    adaptive_path = output_dir / "adaptive_trace.csv"
    constant_path = output_dir / "matched_constant_trace.csv"
    sweep_path = output_dir / "heading_sweep.csv"
    summary_path = output_dir / "summary.json"
    video_path = output_dir / "curvature_taxis_agent.mp4"
    _plot_demo(figure_path, config, adaptive, constant, no_steering)
    _write_trace_csv(adaptive_path, adaptive)
    _write_trace_csv(constant_path, constant)
    _write_sweep_csv(sweep_path, sweep)
    if render_video:
        render_taxis_video(
            video_path,
            config,
            adaptive,
            fps=video_fps,
            playback_speed=video_playback_speed,
        )

    summary: dict[str, Any] = {
        "config": asdict(config),
        "field": {
            "formula": "exp[-0.5(U^2+V^2)-beta*U^4-gamma*U^2*V^2]",
            "non_gaussian": True,
            "non_circular": True,
            "unique_maximum": list(config.source),
        },
        "adaptive": _trace_summary(adaptive),
        "matched_constant": _trace_summary(constant),
        "no_steering": _trace_summary(no_steering),
        "estimator": _estimator_summary(adaptive),
        "speed_relation": _speed_relation_summary(adaptive),
        "heading_sweep": summarize_heading_sweep(sweep),
        "artifacts": {
            "figure": str(figure_path),
            "adaptive_trace": str(adaptive_path),
            "matched_constant_trace": str(constant_path),
            "heading_sweep": str(sweep_path),
            "summary": str(summary_path),
        },
    }
    if render_video:
        summary["artifacts"]["video"] = str(video_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def render_taxis_video(  # noqa: C901, PLR0912, PLR0913, PLR0915
    path: Path,
    config: CurvatureTaxisConfig,
    trace: TaxisTrace | None = None,
    *,
    fps: int = DEFAULT_VIDEO_FPS,
    playback_speed: float = DEFAULT_VIDEO_PLAYBACK_SPEED,
    dpi: int = DEFAULT_VIDEO_DPI,
) -> Path:
    """Render the real adaptive trajectory as an MP4 or animated GIF.

    The arena panel shows the non-Gaussian field, gradient-streamline reference,
    speed-colored trail, heading, locally estimated gradient, and all nine sensor
    positions. The diagnostic panels animate the same sensed curvature, speed,
    confidence, concentration, and distance values stored in ``trace``.

    Parameters
    ----------
    path
        Destination ending in ``.mp4`` or ``.gif``.
    config
        Field, sensing, motor, and speed-response configuration.
    trace
        Existing adaptive trace. When omitted, the deterministic adaptive run is
        generated once and animated directly.
    fps
        Encoded frames per second.
    playback_speed
        Ratio of simulated time to video time. The default 2.0 renders the
        36.96-second default run in about 18.5 seconds, plus a short final hold.
    dpi
        Output resolution scale. The default produces a 1280-by-720 video.
    """
    suffix = path.suffix.lower()
    if suffix not in {".mp4", ".gif"}:
        message = "video path must end in .mp4 or .gif"
        raise ValueError(message)
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        message = "fps must be a positive integer"
        raise ValueError(message)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        message = "dpi must be a positive integer"
        raise ValueError(message)
    if not np.isfinite(playback_speed) or playback_speed <= 0.0:
        message = "playback_speed must be finite and greater than zero"
        raise ValueError(message)

    rendered_trace = trace or run_taxis(config)
    if len(rendered_trace.snapshots) < MINIMUM_TRACE_SNAPSHOTS or not rendered_trace.transitions:
        message = "trace must contain at least one transition"
        raise ValueError(message)
    frame_indices = _video_frame_indices(
        rendered_trace.times,
        fps=fps,
        playback_speed=playback_speed,
    )
    final_hold_frames = max(1, round(VIDEO_END_HOLD_SECONDS * fps))
    frame_indices = np.concatenate(
        (frame_indices, np.full(final_hold_frames, len(rendered_trace.snapshots) - 1)),
    )

    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib import animation
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle

    if suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            message = "FFmpeg is required to render MP4; use a .gif path for Pillow output"
            raise RuntimeError(message)
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            bitrate=DEFAULT_VIDEO_BITRATE_KBPS,
            metadata={
                "title": "Curvature-aware odor navigation",
                "artist": "FMALA elegans simulator",
                "comment": "Rendered from the locally sensed adaptive trajectory",
            },
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    else:
        writer = animation.PillowWriter(
            fps=fps,
            metadata={"title": "Curvature-aware odor navigation"},
        )

    field = config.field()
    coordinates = np.linspace(0.0, config.arena_size, VIDEO_FIELD_GRID_SIZE)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    field_values = np.empty_like(grid_x)
    for row in range(field_values.shape[0]):
        for column in range(field_values.shape[1]):
            field_values[row, column] = field.concentration(
                (grid_x[row, column], grid_y[row, column]),
            )

    stream_coordinates = np.linspace(
        0.05 * config.arena_size,
        0.95 * config.arena_size,
        VIDEO_STREAM_GRID_SIZE,
    )
    stream_x, stream_y = np.meshgrid(stream_coordinates, stream_coordinates)
    gradient_x = np.empty_like(stream_x)
    gradient_y = np.empty_like(stream_y)
    for row in range(gradient_x.shape[0]):
        for column in range(gradient_x.shape[1]):
            gradient, _hessian = field.derivatives(
                (stream_x[row, column], stream_y[row, column]),
            )
            gradient_x[row, column], gradient_y[row, column] = gradient

    times = rendered_trace.times
    transition_times = times[:-1]
    positions = rendered_trace.positions
    speeds = rendered_trace.speeds
    curvatures = np.abs(rendered_trace.estimated_streamline_curvatures)
    distances = rendered_trace.distances
    concentrations = rendered_trace.concentrations
    path_segments = np.stack((positions[:-1], positions[1:]), axis=1)

    figure = plt.figure(figsize=(12.8, 7.2), dpi=dpi, constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.5, 1.0), height_ratios=(1.0, 1.0))
    arena_axis = figure.add_subplot(grid[:, 0])
    response_axis = figure.add_subplot(grid[0, 1])
    history_axis = figure.add_subplot(grid[1, 1])
    history_curvature_axis = history_axis.twinx()

    heatmap = arena_axis.contourf(
        grid_x,
        grid_y,
        field_values,
        levels=np.linspace(0.0, 1.0, 36),
        cmap="viridis",
    )
    figure.colorbar(
        heatmap,
        ax=arena_axis,
        fraction=0.047,
        pad=0.02,
        label="Odor concentration",
    )
    arena_axis.streamplot(
        stream_x,
        stream_y,
        gradient_x,
        gradient_y,
        density=0.72,
        color=(1.0, 1.0, 1.0, 0.28),
        linewidth=0.65,
        arrowsize=0.65,
        zorder=2,
    )
    arena_axis.scatter(
        *config.start,
        color="#25e6e6",
        edgecolor="#111111",
        s=65,
        zorder=7,
        label="Start",
    )
    arena_axis.scatter(
        *config.source,
        marker="*",
        color="#ffe600",
        edgecolor="#111111",
        s=260,
        zorder=8,
        label="Source",
    )
    arena_axis.add_patch(
        Circle(
            config.source,
            config.target_radius,
            fill=False,
            color="#ffe600",
            linewidth=1.6,
            zorder=6,
        ),
    )
    trail = LineCollection(
        [],
        cmap="plasma",
        norm=Normalize(vmin=config.min_speed, vmax=config.max_speed),
        linewidth=4.0,
        zorder=5,
    )
    arena_axis.add_collection(trail)
    agent_marker = arena_axis.scatter(
        [],
        [],
        s=115,
        color="#ffffff",
        edgecolor="#111111",
        linewidth=1.4,
        zorder=10,
        label="Agent",
    )
    sensor_markers = arena_axis.scatter(
        [],
        [],
        s=34,
        c=[],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        edgecolor="#ffffff",
        linewidth=0.5,
        zorder=9,
        label="Nine local sensors",
    )
    (heading_line,) = arena_axis.plot(
        [],
        [],
        color="#ffffff",
        linewidth=2.8,
        zorder=11,
        label="Heading",
    )
    (gradient_line,) = arena_axis.plot(
        [],
        [],
        color="#28d7ff",
        linewidth=2.4,
        zorder=11,
        label="Sensed uphill direction",
    )
    telemetry = arena_axis.text(
        0.025,
        0.975,
        "",
        transform=arena_axis.transAxes,
        va="top",
        ha="left",
        color="#ffffff",
        fontsize=10,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": (0.0, 0.0, 0.0, 0.68),
            "edgecolor": (1.0, 1.0, 1.0, 0.3),
        },
        zorder=12,
    )
    arena_axis.set(
        xlim=(0.0, config.arena_size),
        ylim=(0.0, config.arena_size),
        xlabel="x",
        ylabel="y",
        title="Agent and nine local odor sensors",
        aspect="equal",
    )
    arena_axis.legend(loc="lower right", fontsize=8, framealpha=0.88)

    maximum_curvature = max(1.0, 1.1 * float(np.max(curvatures)))
    curvature_grid = np.linspace(0.0, maximum_curvature, 360)
    full_confidence_response = np.asarray(
        [
            curvature_controlled_speed(
                curvature,
                1.0,
                config.min_speed,
                config.max_speed,
                config.curvature_scale,
                config.speed_exponent,
            )
            for curvature in curvature_grid
        ],
    )
    response_axis.plot(
        curvature_grid,
        full_confidence_response,
        color="#222222",
        linewidth=2.1,
        label="Full-confidence response",
    )
    response_axis.scatter(
        curvatures,
        speeds,
        color="#e9c63b",
        edgecolor="none",
        s=10,
        alpha=0.20,
        label="Experienced samples",
    )
    response_marker = response_axis.scatter(
        [],
        [],
        color="#e45756",
        edgecolor="#ffffff",
        linewidth=0.8,
        s=75,
        zorder=5,
        label="Current command",
    )
    response_axis.set(
        xlim=(0.0, maximum_curvature),
        ylim=(0.92 * config.min_speed, 1.04 * config.max_speed),
        xlabel="|Sensed streamline curvature| (1/unit)",
        ylabel="Forward speed (units/s)",
        title="High curvature continuously slows the agent",
    )
    response_axis.grid(alpha=0.22)
    response_axis.legend(fontsize=8)

    (speed_history,) = history_axis.plot(
        [],
        [],
        color="#e45756",
        linewidth=2.1,
        label="Forward speed",
    )
    (curvature_history,) = history_curvature_axis.plot(
        [],
        [],
        color="#6f4c9b",
        linewidth=1.9,
        label="|Field curvature|",
    )
    speed_history_marker = history_axis.scatter(
        [],
        [],
        color="#e45756",
        edgecolor="#ffffff",
        linewidth=0.7,
        s=45,
        zorder=5,
    )
    curvature_history_marker = history_curvature_axis.scatter(
        [],
        [],
        color="#6f4c9b",
        edgecolor="#ffffff",
        linewidth=0.7,
        s=45,
        zorder=5,
    )
    current_time_line = history_axis.axvline(0.0, color="#333333", linewidth=1.0, alpha=0.6)
    history_axis.set(
        xlim=(0.0, times[-1]),
        ylim=(0.92 * config.min_speed, 1.04 * config.max_speed),
        xlabel="Simulated time (s)",
        ylabel="Forward speed (units/s)",
        title="Sensed geometry and locomotion through time",
    )
    history_curvature_axis.set(
        ylim=(0.0, 1.05 * maximum_curvature),
        ylabel="|Field curvature| (1/unit)",
    )
    history_axis.grid(alpha=0.22)
    history_axis.legend(loc="upper left", fontsize=8)
    history_curvature_axis.legend(loc="upper right", fontsize=8)

    figure.suptitle(
        f"Curvature-aware odor navigation — {playback_speed:g}x playback",
        fontsize=15,
    )

    def update(snapshot_index: int) -> None:
        snapshot = rendered_trace.snapshots[snapshot_index]
        transition_index = min(snapshot_index, len(rendered_trace.transitions) - 1)
        position = np.asarray(snapshot.position, dtype=np.float64)
        forward = np.array(
            [np.cos(snapshot.heading), np.sin(snapshot.heading)],
            dtype=np.float64,
        )
        left = np.array([-forward[1], forward[0]], dtype=np.float64)

        trail.set_segments(path_segments[:snapshot_index].tolist())
        trail.set_array(speeds[:snapshot_index])
        agent_marker.set_offsets(position.reshape(1, 2))
        heading_end = position + 0.42 * forward
        heading_line.set_data(
            (position[0], heading_end[0]),
            (position[1], heading_end[1]),
        )

        gradient_forward, gradient_left = snapshot.observation.geometry.gradient
        gradient_world = gradient_forward * forward + gradient_left * left
        gradient_magnitude = float(np.linalg.norm(gradient_world))
        if gradient_magnitude > 0.0:
            gradient_end = position + 0.55 * gradient_world / gradient_magnitude
        else:
            gradient_end = position
        gradient_line.set_data(
            (position[0], gradient_end[0]),
            (position[1], gradient_end[1]),
        )

        sensor_offsets = np.asarray(
            [
                (0.0, 0.0),
                forward,
                -forward,
                left,
                -left,
                forward + left,
                forward - left,
                -forward + left,
                -forward - left,
            ],
            dtype=np.float64,
        )
        sensor_positions = position + config.sensor_spacing * sensor_offsets
        sensor_values = np.fromiter(
            (getattr(snapshot.observation.stencil, name) for name in STENCIL_NAMES),
            dtype=np.float64,
        )
        sensor_markers.set_offsets(sensor_positions)
        sensor_markers.set_array(sensor_values)

        curvature = curvatures[transition_index]
        speed = speeds[transition_index]
        confidence = snapshot.observation.geometry.confidence
        response_marker.set_offsets(np.array([[curvature, speed]], dtype=np.float64))
        history_stop = transition_index + 1
        speed_history.set_data(transition_times[:history_stop], speeds[:history_stop])
        curvature_history.set_data(
            transition_times[:history_stop],
            curvatures[:history_stop],
        )
        speed_history_marker.set_offsets(
            np.array([[transition_times[transition_index], speed]], dtype=np.float64),
        )
        curvature_history_marker.set_offsets(
            np.array([[transition_times[transition_index], curvature]], dtype=np.float64),
        )
        current_time_line.set_xdata((snapshot.time, snapshot.time))

        completion = ""
        if snapshot_index == len(rendered_trace.snapshots) - 1:
            completion = f"\n{rendered_trace.termination_reason.value.replace('_', ' ').upper()}"
        telemetry.set_text(
            f"t = {snapshot.time:5.2f} s\n"
            f"odor = {concentrations[snapshot_index]:.3f}\n"
            f"|field curvature| = {curvature:.3f} /unit\n"
            f"forward speed = {speed:.3f} units/s\n"
            f"confidence = {confidence:.4f}\n"
            f"distance = {distances[snapshot_index]:.3f} units"
            f"{completion}",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with writer.saving(figure, str(path), dpi=dpi):
            for frame_index in frame_indices:
                update(int(frame_index))
                writer.grab_frame(facecolor=figure.get_facecolor())
    finally:
        plt.close(figure)
    return path


def _video_frame_indices(
    times: NDArray[np.float64],
    *,
    fps: int,
    playback_speed: float,
) -> NDArray[np.int64]:
    """Map fixed-rate video frames to monotonically increasing trace snapshots."""
    if times.ndim != 1 or times.size < MINIMUM_TRACE_SNAPSHOTS or not np.all(np.isfinite(times)):
        message = "times must be a finite one-dimensional array with at least two values"
        raise ValueError(message)
    if np.any(np.diff(times) <= 0.0):
        message = "times must be strictly increasing"
        raise ValueError(message)
    simulated_frame_interval = playback_speed / fps
    frame_times = np.arange(times[0], times[-1], simulated_frame_interval)
    indices = np.searchsorted(times, frame_times, side="left").astype(np.int64)
    indices = np.clip(indices, 0, times.size - 1)
    if indices.size == 0 or indices[-1] != times.size - 1:
        indices = np.append(indices, np.int64(times.size - 1))
    return indices


def _trace_summary(trace: TaxisTrace) -> dict[str, float | bool | str]:
    return {
        "success": trace.success,
        "termination_reason": trace.termination_reason.value,
        "duration": trace.times[-1],
        "steps": len(trace.transitions),
        "initial_distance": trace.distances[0],
        "final_distance": trace.distances[-1],
        "initial_concentration": trace.concentrations[0],
        "final_concentration": trace.concentrations[-1],
        "mean_speed": float(np.mean(trace.speeds)),
        "min_speed": float(np.min(trace.speeds)),
        "max_speed": float(np.max(trace.speeds)),
    }


def _estimator_summary(trace: TaxisTrace) -> dict[str, float]:
    estimated = trace.estimated_streamline_curvatures
    reference = trace.reference_streamline_curvatures
    error = estimated - reference
    return {
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "root_mean_square_error": float(np.sqrt(np.mean(error**2))),
        "correlation": _safe_correlation(estimated, reference),
        "estimated_min": float(np.min(estimated)),
        "estimated_max": float(np.max(estimated)),
        "reference_min": float(np.min(reference)),
        "reference_max": float(np.max(reference)),
        "mean_confidence": float(
            np.mean([item.observation.geometry.confidence for item in trace.transitions]),
        ),
    }


def _speed_relation_summary(trace: TaxisTrace) -> dict[str, float]:
    curvature = np.abs(trace.estimated_streamline_curvatures)
    speeds = trace.speeds
    lower, upper = np.quantile(curvature, [0.25, 0.75])
    low_speeds = speeds[curvature <= lower]
    high_speeds = speeds[curvature >= upper]
    return {
        "curvature_speed_correlation": _safe_correlation(curvature, speeds),
        "low_curvature_quartile_threshold": float(lower),
        "high_curvature_quartile_threshold": float(upper),
        "low_curvature_mean_speed": float(np.mean(low_speeds)),
        "high_curvature_mean_speed": float(np.mean(high_speeds)),
        "speed_difference_low_minus_high": float(np.mean(low_speeds) - np.mean(high_speeds)),
        "curvature_min": float(np.min(curvature)),
        "curvature_max": float(np.max(curvature)),
        "speed_min": float(np.min(speeds)),
        "speed_max": float(np.max(speeds)),
    }


def _write_trace_csv(path: Path, trace: TaxisTrace) -> None:
    base_fields = [
        "time",
        "x",
        "y",
        "heading",
        "center_concentration",
        *[f"sample_{name}" for name in STENCIL_NAMES],
        "gradient_forward",
        "gradient_left",
        "gradient_magnitude",
        "estimated_streamline_curvature",
        "estimated_level_set_curvature",
        "reference_streamline_curvature",
        "reference_level_set_curvature",
        "curvature_confidence",
        "speed",
        "turn_rate",
        "path_curvature",
        "distance_to_source",
        "next_time",
        "next_x",
        "next_y",
        "next_heading",
        *[f"next_sample_{name}" for name in STENCIL_NAMES],
        "next_distance_to_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=base_fields)
        writer.writeheader()
        for index, transition in enumerate(trace.transitions):
            current = trace.snapshots[index]
            following = trace.snapshots[index + 1]
            geometry = transition.observation.geometry
            reference = current.reference_geometry
            row: dict[str, float] = {
                "time": current.time,
                "x": current.position[0],
                "y": current.position[1],
                "heading": current.heading,
                "center_concentration": current.center_concentration,
                "gradient_forward": geometry.gradient[0],
                "gradient_left": geometry.gradient[1],
                "gradient_magnitude": geometry.gradient_magnitude,
                "estimated_streamline_curvature": geometry.streamline_curvature,
                "estimated_level_set_curvature": geometry.level_set_curvature,
                "reference_streamline_curvature": reference.streamline_curvature,
                "reference_level_set_curvature": reference.level_set_curvature,
                "curvature_confidence": geometry.confidence,
                "speed": transition.command.speed,
                "turn_rate": transition.command.turn_rate,
                "path_curvature": transition.command.turn_rate / transition.command.speed,
                "distance_to_source": current.distance_to_source,
                "next_time": following.time,
                "next_x": following.position[0],
                "next_y": following.position[1],
                "next_heading": following.heading,
                "next_distance_to_source": following.distance_to_source,
            }
            for name in STENCIL_NAMES:
                row[f"sample_{name}"] = float(getattr(transition.observation.stencil, name))
                row[f"next_sample_{name}"] = float(
                    getattr(transition.next_observation.stencil, name),
                )
            writer.writerow(row)


def _write_sweep_csv(path: Path, rows: Sequence[HeadingSweepRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _plot_demo(  # noqa: PLR0915 - one cohesive six-panel scientific figure
    path: Path,
    config: CurvatureTaxisConfig,
    adaptive: TaxisTrace,
    constant: TaxisTrace,
    no_steering: TaxisTrace,
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    coordinates = np.linspace(0.0, config.arena_size, 260)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    field = config.field()
    values = np.empty_like(grid_x)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            values[row, column] = field.concentration((grid_x[row, column], grid_y[row, column]))

    figure, axes = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)
    (
        trajectory_axis,
        curvature_axis,
        response_axis,
        distance_axis,
        concentration_axis,
        confidence_axis,
    ) = axes.ravel()
    heatmap = trajectory_axis.contourf(grid_x, grid_y, values, levels=35, cmap="viridis")
    figure.colorbar(heatmap, ax=trajectory_axis, label="Odor concentration")
    trajectory_axis.plot(
        constant.positions[:, 0],
        constant.positions[:, 1],
        "--",
        color="#68b5e8",
        linewidth=2.2,
        label="Matched-mean constant speed",
    )
    trajectory_axis.plot(
        no_steering.positions[:, 0],
        no_steering.positions[:, 1],
        ":",
        color="cyan",
        linewidth=1.8,
        label="No steering",
    )
    points = adaptive.positions.reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    colored_path = LineCollection(segments.tolist(), cmap="plasma", linewidth=4.0)
    colored_path.set_array(adaptive.speeds)
    trajectory_axis.add_collection(colored_path)
    figure.colorbar(colored_path, ax=trajectory_axis, label="Adaptive speed")
    trajectory_axis.plot(
        [],
        [],
        color="#d84a9b",
        linewidth=4.0,
        label="Adaptive (color = speed)",
    )
    trajectory_axis.scatter(*config.start, color="cyan", edgecolor="black", s=70, label="Start")
    trajectory_axis.scatter(
        *config.source,
        marker="*",
        color="yellow",
        edgecolor="black",
        s=230,
        label="Source",
    )
    trajectory_axis.add_patch(
        Circle(config.source, config.target_radius, fill=False, color="yellow"),
    )
    trajectory_axis.set(
        xlim=(0.0, config.arena_size),
        ylim=(0.0, config.arena_size),
        xlabel="x",
        ylabel="y",
        title="Sensed-curvature navigation in a non-Gaussian field",
        aspect="equal",
    )
    trajectory_axis.legend(loc="lower right", fontsize=8)

    curvature_axis.plot(
        adaptive.times[:-1],
        adaptive.estimated_streamline_curvatures,
        label="9-sample estimate",
        color="#e45756",
    )
    curvature_axis.plot(
        adaptive.times[:-1],
        adaptive.reference_streamline_curvatures,
        label="Analytic reference",
        color="black",
        linestyle="--",
    )
    curvature_axis.set(
        xlabel="Time (s)",
        ylabel="Signed curvature (1/unit)",
        title="Odor streamline curvature",
    )
    curvature_axis.grid(alpha=0.25)
    curvature_axis.legend()

    curvature_grid = np.linspace(
        0.0,
        max(1.0, 1.15 * np.max(np.abs(adaptive.estimated_streamline_curvatures))),
        300,
    )
    response = [
        curvature_controlled_speed(
            value,
            1.0,
            config.min_speed,
            config.max_speed,
            config.curvature_scale,
            config.speed_exponent,
        )
        for value in curvature_grid
    ]
    response_axis.plot(
        curvature_grid,
        response,
        color="black",
        label="Full-confidence speed envelope",
    )
    response_axis.scatter(
        np.abs(adaptive.estimated_streamline_curvatures),
        adaptive.speeds,
        color="#e9c63b",
        edgecolor="none",
        s=10,
        alpha=0.55,
        label="Experienced speed (confidence-adjusted)",
    )
    response_axis.set(
        xlabel="|Sensed streamline curvature|",
        ylabel="Speed",
        title="Sensed curvature and confidence set speed",
    )
    response_axis.grid(alpha=0.25)
    response_axis.legend()

    distance_axis.plot(adaptive.times, adaptive.distances, label="Adaptive", color="#f58518")
    distance_axis.plot(
        constant.times,
        constant.distances,
        label="Matched constant",
        color="#4c78a8",
        linestyle="--",
    )
    distance_axis.axhline(config.target_radius, color="black", linestyle=":", label="Target radius")
    distance_axis.set(xlabel="Time (s)", ylabel="Distance", title="Distance to source")
    distance_axis.grid(alpha=0.25)
    distance_axis.legend()

    concentration_axis.plot(
        adaptive.times,
        adaptive.concentrations,
        label="Adaptive",
        color="#54a24b",
    )
    concentration_axis.plot(
        constant.times,
        constant.concentrations,
        label="Matched constant",
        color="#4c78a8",
        linestyle="--",
    )
    concentration_axis.set(xlabel="Time (s)", ylabel="Concentration", title="Experienced odor")
    concentration_axis.grid(alpha=0.25)
    concentration_axis.legend()

    confidence_axis.plot(
        adaptive.times[:-1],
        adaptive.speeds,
        color="#e45756",
        label="Forward speed",
    )
    confidence_axis.set(
        xlabel="Time (s)",
        ylabel="Forward speed",
        title="Locomotion speed and steering are separate commands",
    )
    confidence_axis.grid(alpha=0.25)
    turn_axis = confidence_axis.twinx()
    turn_axis.plot(
        adaptive.times[:-1],
        adaptive.turn_rates,
        color="#4c78a8",
        alpha=0.85,
        label="Motor turn rate",
    )
    turn_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    turn_axis.set(ylabel="Turn rate (rad/s)")
    confidence_axis.legend(loc="upper left")
    turn_axis.legend(loc="upper right")
    figure.suptitle(
        "Local odor geometry controls locomotion speed; steering remains a separate reflex",
        fontsize=15,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _point(position: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    point = np.asarray(position, dtype=np.float64)
    if point.shape != (POSITION_DIMENSIONS,) or not np.all(np.isfinite(point)):
        message = "position must contain two finite coordinates"
        raise ValueError(message)
    return point


def _safe_correlation(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    if (
        first.size < MINIMUM_CORRELATION_SAMPLES
        or np.std(first) <= CONSTANT_TOLERANCE
        or np.std(second) <= CONSTANT_TOLERANCE
    ):
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


__all__ = [
    "DEFAULT_VIDEO_FPS",
    "DEFAULT_VIDEO_PLAYBACK_SPEED",
    "ControlCommand",
    "CurvatureTaxisConfig",
    "CurvatureTaxisEnvironment",
    "CurvedOdorField",
    "HeadingSweepRow",
    "HeadingSweepSummary",
    "TaxisObservation",
    "TaxisSnapshot",
    "TaxisTrace",
    "TaxisTransition",
    "TerminationReason",
    "adaptive_speed",
    "bilateral_turn_rate",
    "render_taxis_video",
    "run_episode",
    "run_heading_sweep",
    "run_matched_constant_speed",
    "run_no_steering",
    "run_taxis",
    "save_demo_artifacts",
    "summarize_heading_sweep",
]
