"""Focused tests for curvature-aware locomotion in the grid simulator."""

from pathlib import Path

import numpy as np
import pytest
from elegans.agent import QuantumNematodeAgent
from elegans.brain.actions import Action
from elegans.brain.arch.mlpreinforce import MLPReinforceBrain, MLPReinforceBrainConfig
from elegans.env import CurvatureNavigationParams, Direction, DynamicForagingEnvironment
from elegans.env.env import ForagingParams
from elegans.utils.config_loader import (
    CurvatureNavigationConfig,
    EnvironmentConfig,
    create_env_from_config,
    load_simulation_config,
)


def curvature_env(*, speed: float = 0.25) -> DynamicForagingEnvironment:
    """Create a small deterministic environment with fixed curvature speed."""
    env = DynamicForagingEnvironment(
        grid_size=20,
        start_pos=(10, 10),
        max_body_length=0,
        foraging=ForagingParams(
            foods_on_grid=0,
            target_foods_to_collect=5,
            gradient_decay_constant=5.0,
            gradient_strength=1.0,
        ),
        curvature_navigation=CurvatureNavigationParams(
            enabled=True,
            min_speed=speed,
            max_speed=speed,
        ),
    )
    env.foods = [(16, 10), (12, 16)]
    return env


def test_default_disabled_movement_is_exactly_one_cell():
    """The opt-in feature must not alter historical movement by default."""
    env = DynamicForagingEnvironment(
        grid_size=20,
        start_pos=(10, 10),
        max_body_length=0,
        foraging=ForagingParams(foods_on_grid=0),
    )

    env.move_agent(Action.LEFT)

    assert env.agent_pos == (9, 10)
    assert env.current_direction is Direction.LEFT
    assert env.translation_occurred
    assert not env.speed_pause_occurred
    assert env.last_field_geometry is None
    assert env.last_locomotion_speed == 1.0


def test_scalar_concentration_gradient_matches_existing_food_vector():
    """The public scalar odor has the simulator's existing vector as its gradient."""
    env = curvature_env()
    env.foods = [(15, 10)]
    point = (10.0, 10.0)
    spacing = 1e-4

    numerical_x = (
        env.get_food_concentration((point[0] + spacing, point[1]))
        - env.get_food_concentration((point[0] - spacing, point[1]))
    ) / (2.0 * spacing)
    numerical_y = (
        env.get_food_concentration((point[0], point[1] + spacing))
        - env.get_food_concentration((point[0], point[1] - spacing))
    ) / (2.0 * spacing)
    vector_x, vector_y = env._compute_food_gradient_vector((10, 10))

    assert (numerical_x, numerical_y) == pytest.approx((vector_x, vector_y), rel=1e-5)


def test_local_stencil_populates_finite_field_geometry_and_speed():
    """Curvature is derived from local odor samples and retained for inspection."""
    env = curvature_env()

    stencil = env.sample_food_odor_stencil()
    geometry = env.sense_food_field_geometry()

    assert stencil.spacing == 1.0
    assert all(
        np.isfinite(value)
        for value in (
            geometry.gradient_magnitude,
            geometry.streamline_curvature,
            geometry.level_set_curvature,
            geometry.confidence,
            env.last_locomotion_speed,
        )
    )
    assert geometry is env.last_field_geometry
    assert 0.0 <= geometry.confidence <= 1.0
    assert env.curvature_navigation.min_speed <= env.last_locomotion_speed <= 1.0


def test_spatially_different_field_curvatures_produce_ordered_speeds():
    """A high-curvature location is slower than a low-curvature location."""
    env = DynamicForagingEnvironment(
        grid_size=30,
        start_pos=(10, 10),
        max_body_length=0,
        foraging=ForagingParams(
            foods_on_grid=0,
            gradient_decay_constant=5.0,
            gradient_strength=1.0,
        ),
        curvature_navigation=CurvatureNavigationParams(
            enabled=True,
            min_speed=0.2,
            max_speed=1.0,
            curvature_scale=0.2,
        ),
    )
    env.foods = [(22, 8), (17, 23), (5, 18)]

    low_geometry = env.sense_food_field_geometry((9.0, 5.0))
    low_speed = env.last_locomotion_speed
    high_geometry = env.sense_food_field_geometry((11.0, 20.0))
    high_speed = env.last_locomotion_speed

    assert abs(high_geometry.streamline_curvature) > abs(low_geometry.streamline_curvature)
    assert high_speed < low_speed


def test_opt_in_policy_gradient_is_derived_from_odor_stencil(monkeypatch):
    """Local policy steering does not call the analytic food-vector oracle."""
    env = curvature_env()
    env.curvature_navigation.use_local_gradient_state = True
    env.foods = [(15, 10)]

    def reject_analytic_vector(_position: tuple[int, ...]) -> tuple[float, float]:
        pytest.fail("analytic food vector must not supply local policy state")

    monkeypatch.setattr(env, "_compute_food_gradient_vector", reject_analytic_vector)
    strength, direction = env.get_navigation_state((10, 10))

    assert strength > 0.0
    assert direction == pytest.approx(0.0, abs=1e-8)


def test_fractional_speed_accumulator_has_expected_move_count():
    """A quarter-cell command produces one grid translation every four ticks."""
    env = curvature_env(speed=0.25)
    start = env.agent_pos

    for _ in range(3):
        env.move_agent(Action.FORWARD)
        assert env.agent_pos == start
        assert env.speed_pause_occurred
        assert not env.translation_occurred

    env.move_agent(Action.FORWARD)

    assert env.agent_pos == (start[0], start[1] + 1)
    assert env.translation_occurred
    assert not env.speed_pause_occurred
    assert env.movement_accumulator == pytest.approx(0.0)


def test_speed_pause_allows_turn_without_translation():
    """Angular response remains active while curvature slows forward motion."""
    env = curvature_env(speed=0.25)

    env.move_agent(Action.LEFT)

    assert env.agent_pos == (10, 10)
    assert env.current_direction is Direction.LEFT
    assert env.speed_pause_occurred
    assert not env.wall_collision_occurred


def test_curvature_movement_preserves_legacy_action_set_validation():
    """Opt-in movement keeps the base environment's explicit action-set error."""
    env = curvature_env()
    env.action_set = [Action.FORWARD]

    with pytest.raises(ValueError, match=r"Action set .* is not supported"):
        env.move_agent(Action.FORWARD)


def test_brain_params_expose_geometry_and_commanded_speed():
    """The brain receives the agent's internal representation of field geometry."""
    env = curvature_env()
    brain = MLPReinforceBrain(
        config=MLPReinforceBrainConfig(),
        input_dim=2,
        num_actions=4,
    )
    agent = QuantumNematodeAgent(brain=brain, env=env, max_body_length=0)

    strength, direction = env.get_state(env.agent_pos)
    params = agent._create_brain_params(strength, direction)

    assert params.odor_field_streamline_curvature is not None
    assert params.odor_field_level_set_curvature is not None
    assert params.odor_curvature_confidence is not None
    assert params.locomotion_speed == pytest.approx(env.last_locomotion_speed)


def test_brain_params_can_use_manyworlds_branch_environment():
    """Copied-branch sensory params must not leak state from the original world."""
    env = curvature_env()
    brain = MLPReinforceBrain(
        config=MLPReinforceBrainConfig(),
        input_dim=2,
        num_actions=4,
    )
    agent = QuantumNematodeAgent(brain=brain, env=env, max_body_length=0)
    branch_env = env.copy()
    branch_env.agent_pos = (3, 4)
    branch_env.current_direction = Direction.LEFT
    branch_env.last_field_geometry = None
    env.last_field_geometry = None

    params = agent._create_brain_params(0.4, 0.2, env=branch_env)

    assert params.agent_position == (3.0, 4.0)
    assert params.agent_direction is Direction.LEFT
    assert branch_env.last_field_geometry is not None
    assert env.last_field_geometry is None


def test_example_config_wires_curvature_navigation_into_environment():
    """The checked-in example exercises the public YAML/factory path."""
    repository_root = Path(__file__).parents[5]
    config = load_simulation_config(
        str(repository_root / "configs/examples/curvature_aware_foraging.yml"),
    )
    assert config.environment is not None
    assert config.brain is not None

    env = create_env_from_config(config.environment, seed=42)

    assert env.curvature_navigation.enabled
    assert env.curvature_navigation.use_local_gradient_state
    assert env.curvature_navigation.min_speed == pytest.approx(0.2)
    assert env.curvature_navigation.max_speed == pytest.approx(1.0)
    assert isinstance(config.brain.config, MLPReinforceBrainConfig)
    assert config.brain.config.use_curvature_features
    assert config.brain.config.curvature_feature_scale == pytest.approx(0.5)


def test_curvature_config_defaults_disabled_and_rejects_excess_grid_speed():
    """Configuration is opt-in and enforces the grid's one-cell speed ceiling."""
    defaults = EnvironmentConfig().get_curvature_navigation_config()
    assert not defaults.enabled
    assert not defaults.use_local_gradient_state

    local_params = CurvatureNavigationConfig(
        enabled=True,
        use_local_gradient_state=True,
    ).to_params()
    assert local_params.use_local_gradient_state

    with pytest.raises(ValueError, match=r"at most 1\.0"):
        CurvatureNavigationConfig(enabled=True, max_speed=1.1).to_params()


def test_agent_reset_preserves_config_and_clears_runtime_state():
    """Episode reset retains opt-in parameters but starts with no movement debt."""
    env = curvature_env(speed=0.25)
    brain = MLPReinforceBrain(
        config=MLPReinforceBrainConfig(),
        input_dim=2,
        num_actions=4,
    )
    agent = QuantumNematodeAgent(brain=brain, env=env, max_body_length=0)
    env.move_agent(Action.FORWARD)
    assert env.movement_accumulator > 0.0
    copied_env = env.copy()
    assert copied_env.curvature_navigation == env.curvature_navigation
    assert copied_env.movement_accumulator == pytest.approx(env.movement_accumulator)

    agent.reset_environment()

    assert agent.env.curvature_navigation.enabled
    assert agent.env.curvature_navigation.min_speed == pytest.approx(0.25)
    assert agent.env.movement_accumulator == 0.0
    assert agent.env.last_field_geometry is None
