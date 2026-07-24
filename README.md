# 🪱 elegans

[![Pre-commit](https://github.com/SyntheticBrains/nematode/workflows/Pre-commit/badge.svg)](https://github.com/SyntheticBrains/nematode/actions/workflows/pre-commit.yml)
[![Tests](https://github.com/SyntheticBrains/nematode/workflows/Tests/badge.svg)](https://github.com/SyntheticBrains/nematode/actions/workflows/tests.yml)
[![Nightly Tests](https://github.com/SyntheticBrains/nematode/workflows/Nightly%20Tests/badge.svg)](https://github.com/SyntheticBrains/nematode/actions/workflows/nightly-tests.yml)
[![codecov](https://codecov.io/gh/SyntheticBrains/nematode/branch/main/graph/badge.svg)](https://codecov.io/gh/SyntheticBrains/nematode)

<p align="center">
  <img src="./docs/assets/images/demo.gif" alt="nematode simulation demo" />
</p>

This project simulates a simplified nematode (C. elegans) navigating dynamic foraging environments to find food while managing satiety, using classical neural network policies as its decision-making brain. It supports a range of reinforcement learning and biologically-inspired architectures for research into bio-inspired sequential decision-making.

## 🧪 Features

- ✅ **Dynamic Foraging Environment**: Realistic multi-food foraging with satiety management and distance efficiency tracking
- ✅ **Predator Evasion**: Multi-objective learning with random-moving predators and gradient-based danger perception
- ✅ **Classical RL Brains**: REINFORCE, PPO, DQN policy/value architectures
- ✅ **Spiking Neural Networks**: Biologically realistic LIF neurons with surrogate gradient learning
- ✅ **Hybrid Architectures**: Reflex + cortex + critic decompositions
- ✅ **Comprehensive Tracking**: Per-run and session-level metrics, plots, and CSV exports
- ✅ **Interactive Workflows**: CLI scripts with flexible configuration
- 🚧 **Expandable Framework**: Modular design for research and experimentation

## 🧠 Brain Architectures

Choose from the following brain architectures:

**Classical (MLP):**

- **MLPPPOBrain** (`mlpppo`): Classical actor-critic with Proximal Policy Optimization (clipped objective, GAE)
- **MLPReinforceBrain** (`mlpreinforce`): Classical multi-layer perceptron with policy gradients (REINFORCE)
- **MLPDQNBrain** (`mlpdqn`): Classical MLP with Deep Q-Network (DQN) learning

**Biologically-Inspired:**

- **SpikingReinforceBrain** (`spikingreinforce`): Biologically realistic spiking neural network with LIF neurons and surrogate gradient learning

**Hybrid:**

- **HybridClassicalBrain** (`hybridclassical`): Small classical MLP reflex + classical cortex MLP + classical critic with mode-gated fusion and a 3-stage curriculum

Select the brain architecture when running simulations:

```bash
uv run ./scripts/run_simulation.py --brain mlpppo            # Best classical (PPO actor-critic)
uv run ./scripts/run_simulation.py --brain mlpreinforce      # Classical REINFORCE
uv run ./scripts/run_simulation.py --brain spikingreinforce  # Biologically realistic (LIF spiking)
uv run ./scripts/run_simulation.py --brain hybridclassical   # Hybrid reflex + cortex + critic
```

## 🚀 Quick Start

### 1. Install Dependencies

Install [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
brew install uv
```

Install the project:

```bash
# Core install with PyTorch and the pixel renderer
uv sync --extra torch --extra pixel

# Docker (with NVIDIA GPU support if available)
docker compose up --build
```

> **Docker GPU Requirements**: For the Docker setup, you need Docker with NVIDIA Container Toolkit installed for GPU acceleration.

### 2. Configure Environment (Optional)

```bash
cp .env.template .env
# Edit .env as needed for local configuration
```

### 3. Run a Simulation

**Command Line Examples:**

```bash
# Classical PPO brain (best classical: actor-critic with GAE)
uv run ./scripts/run_simulation.py --log-level DEBUG --show-last-frame-only --track-per-run --runs 50 --config ./configs/examples/mlpppo_foraging_medium.yml --theme emoji

# Spiking neural network brain (biologically realistic LIF neurons)
uv run ./scripts/run_simulation.py --log-level DEBUG --show-last-frame-only --track-per-run --runs 50 --config ./configs/examples/spikingreinforce_foraging_small.yml --theme emoji

# Hybrid classical brain (reflex + cortex + critic)
uv run ./scripts/run_simulation.py --log-level DEBUG --show-last-frame-only --track-per-run --runs 50 --config ./configs/examples/hybridclassical_foraging_small.yml --theme emoji
```

**Docker Examples:**

```bash
# Run dynamic foraging with MLP brain
docker-compose exec elegans uv run ./scripts/run_simulation.py --log-level DEBUG --show-last-frame-only --track-per-run --runs 50 --config ./configs/examples/mlpreinforce_foraging_medium.yml --theme emoji

# Interactive Docker shell for development
docker-compose exec elegans bash
```

### 4. Run the Curvature-aware Odor Navigation Demo

This deterministic demo uses no learning. A point agent navigates a smooth, non-Gaussian,
non-circular odor field using nine local concentration samples. It reconstructs the local odor
gradient and Hessian, estimates how the uphill odor streamlines bend, and continuously slows in
high-curvature regions while moving faster in low-curvature regions. Motor turn rate, sensed field
curvature, and realized path curvature are logged as separate quantities.

```bash
uv run python scripts/run_curvature_taxis_demo.py --video
```

The command writes a six-panel diagnostic figure, a 1280-by-720 H.264 animation, complete aligned
sensor-transition CSV logs, a 20-heading adaptive-versus-matched-speed comparison, and a JSON
summary to `exports/curvature_taxis/`. See the
[curvature-navigation study](docs/research/curvature_navigation/README.md) for the equations,
interpretation guide, simulator port, and experiment plan.
MP4 rendering requires FFmpeg; omit `--video` to generate the other artifacts without it.

To enable the same curvature estimator and speed law in the main discrete *C. elegans* foraging
simulator, run:

```bash
uv run scripts/run_simulation.py \
  --config configs/examples/curvature_aware_foraging.yml
```

The example reconstructs its food-gradient steering signal from the same local odor stencil and
adds signed curvature plus estimator confidence to the MLP policy input. The simulator feature is
opt-in; existing configurations retain their original two-input policy and one-cell-per-action
movement exactly.

### 5. Run the Sensorimotor-Influence Toy Study

This reward-free scalar experiment tests whether an online action-aware predictor outperforms a
matched action-blind predictor, then uses their loss difference to gate a fixed controller's vigor.
It includes cable-off, matched-yoked, probe-withdrawal, reversed-action, and gain/noise controls.

```bash
uv run ./scripts/run_sensorimotor_influence.py \
  --output artifacts/sensorimotor_influence
```

See the [study overview](docs/research/sensorimotor_influence/README.md) for the claim boundary,
analytic target, validation protocol, and quick-run command.

## ❓ How It Works

### Dynamic Foraging Environment

1. **State Perception**: The nematode perceives its environment through a viewport (distance to nearest food, gradient information, satiety level)
2. **Brain Processing**: The selected brain architecture processes the state
3. **Action Selection**: Brain outputs action probabilities (forward, left, right, stay)
4. **Environment Update**: Agent moves, satiety decays, and receives reward signal
5. **Food Collection**: When reaching food, satiety is restored and new food spawns
6. **Learning**: Brain parameters are updated based on reward feedback
7. **Repeat**: Process continues until all foods are collected, satiety reaches zero (starvation), or maximum steps reached

### Spiking Neural Network

The spiking brain architecture provides biologically realistic neural computation with modern gradient-based learning:

- **Leaky Integrate-and-Fire (LIF) Neurons**: Membrane potential dynamics with spike generation
- **Surrogate Gradient Descent**: Differentiable spike approximation enabling backpropagation
- **Policy Gradient Learning (REINFORCE)**: Same proven algorithm as the MLP brains
- **Population Coding**: Gaussian tuning curves for improved input discrimination

**Key Features:**

- Biologically plausible temporal dynamics with LIF neurons
- Effective gradient-based learning through surrogate gradients
- Configurable network architecture (timesteps, hidden layers, hidden size)
- Achieves 100% success on foraging tasks, 63% on predator evasion

### Predator Evasion

The predator evasion system adds a challenging multi-objective learning task where agents must balance food collection with survival:

**Predator Mechanics:**

- Random movement patterns with configurable speed (default 1 unit/step)
- Detection radius (default 8 units) creating danger zones
- Kill radius (default 0 units) for lethal collisions
- Multiple predators with independent movement

**Gradient-Based Perception:**

- **Food gradients**: Attractive exponential decay guiding agents toward food
- **Predator gradients**: Repulsive exponential decay warning of danger
- **Gradient superposition**: Combined vector field for multi-objective decision-making
- Agent perceives both food and threat locations through unified gradient system

**Learning Dynamics:**

- **Proximity penalty**: Continuous negative reward when in danger zone (detection radius)
- **Death penalty**: Large negative reward (default -10.0) on predator collision
- **Multi-objective optimization**: Agents learn to collect food while avoiding threats
- **Predator metrics**: Track encounters, successful evasions, and survival strategies

## 🏆 Top Benchmarks

Track and compare performance across different brain architectures and optimization strategies. The benchmark system helps identify effective approaches and advances the state-of-the-art in bio-inspired navigation.

### Quick Start with Benchmarks

```bash
# Run 10+ independent training sessions
for session in {1..10}; do
    uv run scripts/run_simulation.py \
        --config configs/your_config.yml \
        --track-experiment \
        --runs 50
done

# Submit all sessions together
uv run scripts/benchmark_submit.py \
    --experiments experiments/* \
    --category foraging_small/classical \
    --contributor "Your Name"

# Regenerate leaderboards
uv run scripts/benchmark_submit.py regenerate
```

### Current Leaders

#### Foraging Small - Classical

| Brain | Score | Success Rate | Learning Speed | Stability | Distance Efficiency | Sessions | Contributor | Date |
|---|---|---|---|---|---|---|---|---|
| mlpppo | 0.835 ± 0.007 | 96.7% ± 1.3% | 0.93 ± 0.01 | 0.95 ± 0.05 | 0.47 ± 0.02 | 12 | @chrisjz | 2025-12-28 |
| mlpreinforce | 0.810 ± 0.014 | 95.1% ± 1.9% | 0.91 ± 0.02 | 0.99 ± 0.03 | 0.39 ± 0.04 | 12 | @chrisjz | 2025-12-29 |

#### Predator Small - Classical

| Brain | Score | Success Rate | Learning Speed | Stability | Distance Efficiency | Sessions | Contributor | Date |
|---|---|---|---|---|---|---|---|---|
| mlpppo | 0.728 ± 0.029 | 83.3% ± 2.9% | 0.92 ± 0.02 | 0.62 ± 0.05 | 0.51 ± 0.02 | 12 | @chrisjz | 2025-12-29 |
| mlpreinforce | 0.624 ± 0.123 | 73.4% ± 10.9% | 0.84 ± 0.09 | 0.52 ± 0.19 | 0.39 ± 0.07 | 12 | @chrisjz | 2025-12-29 |

See [BENCHMARKS.md](BENCHMARKS.md) for complete leaderboards and submission guidelines.

## 📊 Simulation Visualization

The default Pixel theme renders the simulation in a Pygame window with biologically accurate sprites inspired by real *C. elegans* ecology:

![Pixel Theme](docs/assets/images/pixel_theme.png)

### Entities

| Entity | Visual | Biological Basis |
|--------|--------|-----------------|
| **Nematode head** | Translucent rounded head with pharynx bulb, directional facing | *C. elegans* head morphology |
| **Nematode body** | Connected tan/cream segments with tapered tail | *C. elegans* body coloring |
| **Food** | Green clustered dots | *E. coli* / OP50 bacterial lawns |
| **Random predator** | Purple branching tendrils | Nematode-trapping fungi (*Arthrobotrys oligospora*) |
| **Stationary predator** | Purple ring/net structure with toxic zone | Constricting ring traps (*Drechslerella*) |
| **Pursuit predator** | Orange-red arachnid shape | Predatory mites |

### Environment Layers

| Layer | Description |
|-------|-------------|
| **Soil** | Dark earth background with subtle texture |
| **Temperature zones** | Blue (cold) through neutral to red/orange (hot) overlays based on thermal gradient |
| **Toxic zones** | Purple overlay around stationary predators indicating damage radius |

### Status Bar

The status bar displays session-level information (run progress, cumulative wins, total food eaten, average steps) and run-level information (current step, food collected, health, satiety, danger status, temperature zone).

### Alternative Themes

Console-based themes (ASCII, Emoji, Rich, etc.) are also available for headless or CI environments. Set `--theme` when running the simulation to switch between them.

### Session Summary

After all runs complete, a summary report is printed to the console:

```text
Total runs completed: 50
Successful runs: 30 (60.0%)
Failed runs - Starved: 2 (4.0%)
Failed runs - Health Depleted: 15 (30.0%)
Failed runs - Max Steps: 3 (6.0%)
Average foods collected per run: 8.18
Average steps per run: 300.20
Average reward per run: 1.93
Average distance efficiency: 0.32
Average survival score: 0.72
Average temperature comfort: 0.68
Success rate: 60.00%
```

## 🧰 Built With

- **[PyTorch](https://pytorch.org/)**: Classical neural networks
- **[uv](https://github.com/astral-sh/uv)**: Modern Python dependency management
- **[Pydantic](https://pydantic.dev/)**: Data validation and settings
- **[Rich](https://rich.readthedocs.io/)**: Beautiful terminal output

## 🔬 Research Applications

This project serves as a platform for exploring:

- **Bio-inspired RL**: Reinforcement learning in ecologically-valid foraging environments
- **Biological Modeling**: Simplified models of neural decision-making
- **Hybrid Architectures**: Reflex/cortex/critic decompositions and modular policies
- **Spiking Networks**: Surrogate gradient learning in biologically realistic neurons

## 🗺️ Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the comprehensive project roadmap.

### Upcoming Features

- **SOTA RL Baselines**: Modern algorithms (SAC, TD3) for credible classical comparison
- **Enhanced Sensory Systems**: Thermotaxis, oxygen sensing, mechanosensation (touch response), and health/damage systems
- **Advanced Predator Behaviors**: Stationary traps, pursuit patterns, patrol routes, and group hunting strategies
- **Architecture Analysis**: Ablation studies, interpretability tools, and systematic feature importance ranking
- **Learning & Memory**: Associative learning systems (STAM, ITAM, LTAM) with biological timescales
- **Evolution & Breeding**: Genetic algorithms, Baldwin effect, co-evolution of predators and prey
- **Multi-Agent Scenarios**: Cooperative and competitive foraging with pheromone communication and emergent behaviors
- **Real-World Validation**: WormBot deployment, C. elegans lab collaborations, cross-organism transfer (Drosophila, zebrafish)

### Research Applications

This platform enables research in:

- Bio-inspired reinforcement learning on biologically-relevant navigation tasks
- Multi-objective decision-making (foraging vs. survival)
- Comparative analysis of classical and spiking neural architectures
- Hybrid reflex/cortex computation in ecologically-valid environments
- Universal computational principles transferable across organisms (C. elegans → Drosophila → zebrafish) and domains (foraging → robotics)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for complete development setup instructions, code style guidelines, testing procedures, and pull request process.

### Areas We Need Help With

- **Foraging Environment Extensions**: Social behaviors, food quality, temperature gradients
- **Multi-Agent Scenarios**: Cooperative and competitive foraging dynamics
- **Visualization Tools**: Real-time learning analysis and environment rendering
- **Documentation**: Tutorials and examples for dynamic environments
- **Testing**: Performance benchmarks and foraging strategy analysis

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **[OpenSpec](https://github.com/Fission-AI/OpenSpec)**: For providing the OpenSpec framework for structured, spec-driven AI development
- **C. elegans Research Community**: For inspiring this computational model
