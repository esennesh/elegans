"""Unit tests for the MLP brain architecture."""

# pyright: reportPrivateImportUsage=false

from typing import get_args, get_type_hints

import numpy as np
import pytest
import torch
from elegans.brain.actions import Action, ActionData
from elegans.brain.arch import BrainParams
from elegans.brain.arch.dtypes import DeviceType
from elegans.brain.arch.mlpreinforce import MLPReinforceBrain, MLPReinforceBrainConfig
from elegans.env import Direction


class TestMLPReinforceBrainConfig:
    """Test cases for MLP brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MLPReinforceBrainConfig()

        assert config.baseline == 0.0
        assert config.baseline_alpha == 0.05
        assert config.entropy_beta == 0.01
        assert config.gamma == 0.99
        assert config.hidden_dim == 64
        assert config.learning_rate == 0.01
        assert config.lr_scheduler_step_size == 100
        assert config.lr_scheduler_gamma == 0.9
        assert config.num_hidden_layers == 2
        assert not config.use_curvature_features
        assert config.curvature_feature_scale == 0.5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MLPReinforceBrainConfig(
            hidden_dim=128,
            learning_rate=0.001,
            gamma=0.95,
            num_hidden_layers=3,
        )

        assert config.hidden_dim == 128
        assert config.learning_rate == 0.001
        assert config.gamma == 0.95
        assert config.num_hidden_layers == 3

    @pytest.mark.parametrize("scale", [0.0, -0.1])
    def test_curvature_feature_scale_must_be_positive(self, scale):
        """Curvature normalization cannot use a zero or negative scale."""
        with pytest.raises(ValueError, match="greater than 0"):
            MLPReinforceBrainConfig(curvature_feature_scale=scale)


class TestMLPReinforceBrain:
    """Test cases for the MLP brain architecture."""

    @pytest.fixture
    def config(self) -> MLPReinforceBrainConfig:
        """Create a test configuration."""
        return MLPReinforceBrainConfig(
            hidden_dim=32,
            learning_rate=0.01,
            num_hidden_layers=2,
        )

    @pytest.fixture
    def brain(self, config) -> MLPReinforceBrain:
        """Create a test MLP brain."""
        return MLPReinforceBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
            lr_scheduler=False,  # Disable for simpler testing
        )

    def test_brain_initialization(self, brain, config):
        """Test MLP brain initialization."""
        assert brain.input_dim == 2
        assert brain.num_actions == 4
        assert brain.entropy_beta == config.entropy_beta
        assert brain.gamma == config.gamma
        assert brain.baseline == config.baseline
        assert brain.training is True

        # Check policy network exists
        assert brain.policy is not None
        assert isinstance(brain.policy, torch.nn.Sequential)

    def test_build_network(self, brain):
        """Test network architecture building."""
        # Network should have correct input/output dimensions
        test_input = torch.randn(1, brain.input_dim)
        output = brain.policy(test_input)
        assert output.shape == (1, brain.num_actions)

    def test_preprocess(self, brain):
        """Test state preprocessing."""
        params = BrainParams(
            gradient_strength=0.8,
            gradient_direction=1.5,
            agent_position=(2, 3),
            agent_direction=Direction.UP,
        )

        features = brain.preprocess(params)
        assert isinstance(features, np.ndarray)
        assert len(features) == 2  # gradient_strength, relative_angle
        assert features.dtype == np.float32
        assert -1.0 <= features[1] <= 1.0  # Normalized relative angle

    def test_preprocess_none_values(self, brain):
        """Test preprocessing with None values."""
        params = BrainParams()
        features = brain.preprocess(params)

        assert len(features) == 2
        assert features[0] == 0.0  # gradient_strength defaults to 0

    def test_legacy_preprocess_ignores_curvature_fields(self, brain):
        """The default policy retains its historical two-feature input exactly."""
        params = BrainParams(
            gradient_strength=0.8,
            gradient_direction=np.pi / 2,
            agent_direction=Direction.UP,
            odor_field_streamline_curvature=3.0,
            odor_curvature_confidence=0.75,
        )

        features = brain.preprocess(params)

        assert features == pytest.approx(np.array([0.8, 0.0], dtype=np.float32))
        assert brain.policy[0].in_features == 2

    def test_curvature_features_extend_preprocessing_and_network_input(self):
        """Opt-in policies receive bounded curvature and confidence features."""
        config = MLPReinforceBrainConfig(
            hidden_dim=16,
            use_curvature_features=True,
            curvature_feature_scale=0.5,
        )
        brain = MLPReinforceBrain(
            config=config,
            input_dim=4,
            num_actions=4,
            device=DeviceType.CPU,
            lr_scheduler=False,
        )
        params = BrainParams(
            gradient_strength=0.8,
            gradient_direction=np.pi / 2,
            agent_direction=Direction.UP,
            odor_field_streamline_curvature=0.25,
            odor_curvature_confidence=0.75,
        )

        features = brain.preprocess(params)

        assert features == pytest.approx(
            np.array([0.8, 0.0, np.tanh(0.5), 0.75], dtype=np.float32),
        )
        assert brain.input_dim == 4
        assert brain.policy[0].in_features == 4
        assert brain.forward(features).shape == (4,)

    def test_curvature_features_default_to_zero_when_sensing_is_unavailable(self):
        """The opt-in preprocessing path remains finite before a sensor is populated."""
        config = MLPReinforceBrainConfig(use_curvature_features=True)
        brain = MLPReinforceBrain(
            config=config,
            input_dim=4,
            num_actions=4,
            device=DeviceType.CPU,
            lr_scheduler=False,
        )

        features = brain.preprocess(BrainParams())

        assert features.shape == (4,)
        assert features[2:] == pytest.approx((0.0, 0.0))

    def test_constructor_rejects_input_dimension_inconsistent_with_config(self):
        """Misconfigured models fail early instead of at their first forward pass."""
        config = MLPReinforceBrainConfig(use_curvature_features=True)

        with pytest.raises(ValueError, match="input_dim must be 4"):
            MLPReinforceBrain(config=config, input_dim=2, num_actions=4)

    def test_legacy_constructor_still_accepts_custom_input_dimension(self):
        """Curvature opt-in must not narrow the historical constructor API."""
        brain = MLPReinforceBrain(
            config=MLPReinforceBrainConfig(),
            input_dim=3,
            num_actions=4,
        )

        assert brain.input_dim == 3
        assert brain.policy[0].in_features == 3

    def test_factory_builds_four_input_policy_when_curvature_is_enabled(self):
        """The configured production factory derives the opt-in input dimension."""
        from elegans.brain.arch.dtypes import BrainType
        from elegans.optimizers.learning_rate import ConstantLearningRate
        from elegans.utils.brain_factory import setup_brain_model
        from elegans.utils.config_loader import ParameterInitializerConfig

        brain = setup_brain_model(
            brain_type=BrainType.MLP_REINFORCE,
            brain_config=MLPReinforceBrainConfig(use_curvature_features=True),
            device=DeviceType.CPU,
            learning_rate=ConstantLearningRate(learning_rate=0.01),
            parameter_initializer_config=ParameterInitializerConfig(),
        )

        assert isinstance(brain, MLPReinforceBrain)
        assert brain.input_dim == 4
        assert brain.policy[0].in_features == 4

    def test_factory_type_hints_remain_runtime_resolvable(self):
        """Public factory annotations support normal runtime introspection."""
        from elegans.optimizers.learning_rate import ConstantLearningRate
        from elegans.utils.brain_factory import setup_brain_model

        hints = get_type_hints(setup_brain_model)

        assert ConstantLearningRate in get_args(hints["learning_rate"])

    def test_forward(self, brain):
        """Test forward pass through the network."""
        x = np.array([0.5, 0.3], dtype=np.float32)
        logits = brain.forward(x)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (brain.num_actions,)
        assert torch.all(torch.isfinite(logits))

    def test_run_brain(self, brain):
        """Test running the brain for decision making."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        actions = brain.run_brain(params, top_only=True, top_randomize=True)

        assert len(actions) == 1
        action_data = actions[0]
        assert isinstance(action_data, ActionData)
        assert action_data.action in [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY]
        assert 0.0 <= action_data.probability <= 1.0

        # Check that episode data is being tracked
        assert len(brain.episode_states) > 0
        assert len(brain.episode_actions) > 0
        assert len(brain.episode_log_probs) > 0

    def test_compute_discounted_return(self, brain):
        """Test discounted return computation."""
        rewards = [1.0, 0.5, 0.2]
        gamma = 0.9

        discounted_return = brain.compute_discounted_return(rewards, gamma)

        expected = 1.0 + 0.9 * 0.5 + 0.81 * 0.2
        assert np.isclose(discounted_return, expected)

    def test_learn(self, brain):
        """Test learning with policy gradient."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        # Run brain to generate action
        brain.run_brain(params, top_only=True, top_randomize=False)

        # Learn from multiple steps
        for _ in range(5):
            brain.learn(params, reward=0.5, episode_done=False)

        # Trigger update
        brain.learn(params, reward=1.0, episode_done=True)

        # Weights should have changed (unless gradients were zero by chance)
        # Check that learning doesn't crash and weights remain finite
        for param in brain.policy.parameters():
            assert torch.all(torch.isfinite(param))

    def test_episode_buffer_management(self, brain):
        """Test episode buffer is managed correctly."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Run multiple steps
        for _ in range(3):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Buffer should have data
        assert len(brain.episode_states) > 0
        assert len(brain.episode_rewards) > 0

        # Complete episode
        brain.learn(params, reward=1.0, episode_done=True)

        # Buffer should be cleared
        assert len(brain.episode_states) == 0
        assert len(brain.episode_rewards) == 0
        assert len(brain.episode_actions) == 0
        assert len(brain.episode_log_probs) == 0

    def test_update_memory(self, brain):
        """Test memory update functionality."""
        # Should be a no-op for MLP brain
        brain.update_memory(reward=0.5)
        # Just verify it doesn't crash

    def test_post_process_episode(self, brain):
        """Test episode post-processing."""
        # post_process_episode is a no-op for MLPReinforceBrain
        brain.post_process_episode()
        # Just verify it doesn't crash

    def test_action_set_property(self, brain):
        """Test action_set property."""
        action_set = brain.action_set
        assert len(action_set) == 4
        assert Action.FORWARD in action_set
        assert Action.LEFT in action_set
        assert Action.RIGHT in action_set
        assert Action.STAY in action_set

    def test_build_brain_not_implemented(self, brain):
        """Test that build_brain raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            brain.build_brain()

    def test_copy_not_implemented(self, brain):
        """Test that copy raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            brain.copy()


class TestMLPReinforceBrainIntegration:
    """Integration tests for MLP brain with full simulation workflow."""

    def test_full_episode_workflow(self):
        """Test a complete episode workflow."""
        config = MLPReinforceBrainConfig(hidden_dim=16, learning_rate=0.01)
        brain = MLPReinforceBrain(
            config=config,
            input_dim=2,
            num_actions=4,
            device=DeviceType.CPU,
            lr_scheduler=False,
        )

        # Simulate multiple steps in an episode
        episode_rewards = []
        rng = np.random.default_rng(42)

        for step in range(10):
            params = BrainParams(
                gradient_strength=rng.random(),
                gradient_direction=rng.random() * 2 * np.pi,
                agent_position=(step, step),
                agent_direction=Direction.UP,
            )

            # Run brain
            actions = brain.run_brain(params, top_only=True, top_randomize=False)
            assert len(actions) == 1

            # Learn
            reward = rng.random() - 0.5  # Random reward
            brain.learn(params, reward, episode_done=(step == 9))
            episode_rewards.append(reward)

        # Post-process episode
        brain.post_process_episode()

        # Check that episode completed successfully
        assert len(brain.history_data.rewards) == 10

    def test_training_vs_evaluation_mode(self):
        """Test behavior differences between training and evaluation."""
        config = MLPReinforceBrainConfig(hidden_dim=16)
        brain = MLPReinforceBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Training mode
        brain.training = True
        actions_train = brain.run_brain(params, top_only=True, top_randomize=False)

        # Evaluation mode
        brain.training = False
        actions_eval = brain.run_brain(params, top_only=True, top_randomize=False)

        # Both should return valid actions
        assert len(actions_train) == 1
        assert len(actions_eval) == 1

    def test_baseline_updates(self):
        """Test that baseline is updated during learning."""
        config = MLPReinforceBrainConfig(baseline=0.0, baseline_alpha=0.1)
        brain = MLPReinforceBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Initial baseline
        initial_baseline = brain.baseline

        # Run multiple episodes
        for _ in range(5):
            brain.run_brain(params, top_only=True, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=True)

        # Baseline should have updated
        assert brain.baseline != initial_baseline

    def test_exploration_noise(self):
        """Test that exploration noise is added during training."""
        config = MLPReinforceBrainConfig(hidden_dim=16)
        brain = MLPReinforceBrain(config=config, input_dim=2, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        # Set seed for reproducibility
        torch.manual_seed(42)
        brain.training = True
        actions1 = brain.run_brain(params, top_only=True, top_randomize=False)

        # Different seed should give different results due to noise
        torch.manual_seed(43)
        actions2 = brain.run_brain(params, top_only=True, top_randomize=False)

        # Actions might be the same, but probabilities should potentially differ
        # This test mainly ensures noise addition doesn't crash
        assert actions1 is not None
        assert actions2 is not None
