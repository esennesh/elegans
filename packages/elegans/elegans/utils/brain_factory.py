"""Factory for creating brain instances from configuration.

Consolidates brain instantiation logic used across entrypoint scripts
(run_simulation.py, run_evolution.py) into a single reusable module.
"""

from __future__ import annotations

from elegans.brain.arch import (
    Brain,
    HybridClassicalBrainConfig,
    MLPDQNBrainConfig,
    MLPPPOBrainConfig,
    MLPReinforceBrainConfig,
    SpikingReinforceBrainConfig,
)
from elegans.brain.arch.dtypes import BrainType, DeviceType
from elegans.logging_config import logger
from elegans.optimizers.learning_rate import (  # noqa: TC001 - public runtime annotations
    AdamLearningRate,
    ConstantLearningRate,
    DynamicLearningRate,
    PerformanceBasedLearningRate,
)
from elegans.utils.config_loader import (
    ParameterInitializerConfig,
    create_parameter_initializer_instance,
)


def setup_brain_model(  # noqa: C901
    brain_type: BrainType,
    brain_config: MLPReinforceBrainConfig
    | MLPPPOBrainConfig
    | MLPDQNBrainConfig
    | HybridClassicalBrainConfig
    | SpikingReinforceBrainConfig,
    device: DeviceType,
    learning_rate: ConstantLearningRate
    | DynamicLearningRate
    | AdamLearningRate
    | PerformanceBasedLearningRate,
    parameter_initializer_config: ParameterInitializerConfig,
) -> Brain:
    """Set up the brain model based on the specified brain type.

    Args:
        brain_type: The type of brain architecture to use.
        brain_config: Configuration for the brain architecture.
        device: The device to use for simulation.
        learning_rate: The learning rate configuration for the brain.
        parameter_initializer_config: Configuration for parameter initialization.

    Returns
    -------
        Brain: An instance of the selected brain model.

    Raises
    ------
        ValueError: If an unknown brain type is provided.
    """
    del learning_rate  # Kept in the public factory API for caller compatibility.

    if brain_type in (BrainType.MLP_REINFORCE, BrainType.MLP):
        from elegans.brain.arch.mlpreinforce import MLPReinforceBrain

        if not isinstance(brain_config, MLPReinforceBrainConfig):
            error_message = (
                "The 'mlpreinforce' brain architecture requires an MLPReinforceBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        input_dim = 4 if brain_config.use_curvature_features else 2

        brain = MLPReinforceBrain(
            config=brain_config,
            input_dim=input_dim,
            num_actions=4,
            lr_scheduler=True,
            device=device,
            parameter_initializer=parameter_initializer,
        )
    elif brain_type in (BrainType.MLP_PPO, BrainType.PPO):
        from elegans.brain.arch.mlpppo import MLPPPOBrain

        if not isinstance(brain_config, MLPPPOBrainConfig):
            error_message = (
                "The 'mlpppo' brain architecture requires a MLPPPOBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = MLPPPOBrain(
            config=brain_config,
            input_dim=2,
            num_actions=4,
            device=device,
            parameter_initializer=parameter_initializer,
        )
    elif brain_type == BrainType.MLP_DQN:
        from elegans.brain.arch.mlpdqn import MLPDQNBrain

        if not isinstance(brain_config, MLPDQNBrainConfig):
            error_message = (
                "The 'mlpdqn' brain architecture requires a MLPDQNBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = MLPDQNBrain(
            config=brain_config,
            input_dim=2,
            num_actions=4,
            device=device,
            parameter_initializer=parameter_initializer,
        )
    elif brain_type in (BrainType.SPIKING_REINFORCE, BrainType.SPIKING):
        from elegans.brain.arch.spikingreinforce import SpikingReinforceBrain

        if not isinstance(brain_config, SpikingReinforceBrainConfig):
            error_message = (
                "The 'spikingreinforce' brain architecture requires a SpikingReinforceBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        input_dim = 4 if brain_config.use_separated_gradients else 2

        brain = SpikingReinforceBrain(
            config=brain_config,
            input_dim=input_dim,
            num_actions=4,
            device=device,
        )
    elif brain_type == BrainType.HYBRID_CLASSICAL:
        from elegans.brain.arch.hybridclassical import HybridClassicalBrain

        if not isinstance(brain_config, HybridClassicalBrainConfig):
            error_message = (
                "The 'hybridclassical' brain architecture requires a "
                "HybridClassicalBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        brain = HybridClassicalBrain(
            config=brain_config,
            num_actions=4,
            device=device,
        )
    else:
        error_message = f"Unknown brain type: {brain_type}"
        logger.error(error_message)
        raise ValueError(error_message)

    return brain
