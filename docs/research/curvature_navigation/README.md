# Curvature-aware odor navigation

This study tests a specific locomotion hypothesis:

> An agent can estimate the local geometry of an odor field from concentration
> samples and continuously reduce forward speed where the field's uphill flow
> bends sharply.

The implementation deliberately separates three quantities that were previously
called “curvature”:

- **Odor-field streamline curvature** describes how the normalized uphill odor
  vector changes across space. This is the signal used to regulate speed.
- **Motor turn rate** is the controller's left/right angular command in radians
  per second.
- **Path curvature** is motor turn rate divided by forward speed. It describes
  the agent's realized trajectory, not its sensory estimate.

The toy experiment uses a rotated anisotropic quartic field with one unique
source. Its contours are neither Gaussian nor circular, and its streamline
curvature varies along the route. The controller receives no source coordinates
or analytic derivatives. It senses nine nearby concentrations, estimates a local
gradient and Hessian, derives field curvature, and applies a bounded continuous
speed law.

The same estimator and speed law are also available in the main discrete
`DynamicForagingEnvironment`. That integration is opt-in so existing agents and
checkpoints retain their original one-cell-per-action behavior. The checked-in
example also opts into stencil-derived steering and adds signed streamline
curvature plus estimator confidence to the MLP REINFORCE policy inputs. Thus its
food-navigation channel no longer receives the analytic food-coordinate vector.
Predator repulsion remains on the simulator's existing vector sensor because a
scalar predator-odor field has not been defined.

## Reproduce

From the repository root:

```bash
# Continuous non-Gaussian toy experiment
uv run python scripts/run_curvature_taxis_demo.py \
  --output-dir exports/curvature_navigation \
  --video

# Main simulator with curvature-aware locomotion enabled
uv run scripts/run_simulation.py \
  --config configs/examples/curvature_aware_foraging.yml
```

The toy command writes the complete sensor/transition trace, heading sweep,
machine-readable summary, diagnostic figure, and H.264 animation. The video
shows the real speed-colored trajectory, heading, nine sensor locations, sensed
uphill direction, curvature-speed response, and time histories. Curated results
are preserved under `artifacts/curvature_navigation/`. The simulator
configuration uses the identical geometry estimator and speed response on the
multi-food odor field, feeds its locally estimated gradient to the steering
policy, and exposes the curvature estimate as an optional learnable policy
feature.

The MP4 writer requires FFmpeg on the system path. Without FFmpeg, omit
`--video`; the figure, traces, sweep, and summary do not depend on it.

## Documents

- [Methods memo](methods_memo.md)
- [Experiment plan](experiment_plan.md)
- [Results and interpretation](results.md)
- [Curated figure and machine-readable results](../../../artifacts/curvature_navigation/README.md)

## Claim boundary

This establishes a computational mechanism in simulation: local concentration
samples are sufficient to estimate field geometry, and that estimate can
continuously modulate locomotor speed. It does not establish that *C. elegans*
uses this exact nine-point stencil, that the chosen speed law is biologically
identified, or that speed modulation improves fitness in every odor landscape.
Those are separate empirical questions.
