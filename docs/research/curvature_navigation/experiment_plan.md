# Experiment plan

The implementation supports a staged validation rather than treating one
successful trajectory as sufficient evidence.

## 1. Estimator validation

- Compare sensed finite-difference curvature with analytic curvature throughout
  the non-Gaussian field, excluding only points with explicitly low confidence.
- Repeat after rotating and translating the field to verify coordinate
  invariance.
- Confirm that linear fields have zero streamline curvature and that critical
  points produce finite, low-confidence outputs.
- Halve sensor spacing and check for the expected reduction in noiseless
  finite-difference error until numerical precision dominates.

## 2. Motor-law validation

- Verify the exact minimum, maximum, and half-response speeds.
- Check that speed decreases monotonically with curvature magnitude.
- Compare low- and high-curvature trajectory quartiles and require lower mean
  speed in the high-curvature quartile.
- Plot observed speeds over the analytic response curve so clipping or wiring
  mistakes are visible.

## 3. Behavioral ablations

- Adaptive speed versus a constant-speed controller matched to the adaptive
  run's mean speed.
- Correct sensed curvature versus a shuffled curvature time series.
- Nine-point spatial sensing versus a temporally accumulated head-sweep
  estimator.
- Streamline-curvature control versus level-set-curvature control.

Use identical starting states and steering logic. Outcomes should include source
success, time, path length, integrated heading error, concentration exposure,
and curvature-estimation error.

## 4. Main-simulator study

Enable curvature navigation in the multi-food foraging environment and compare
it with the default fixed-speed dynamics across held-out seeds. Prespecify:

- curvature-scale and speed-range sweeps;
- food density and spatial clustering;
- gradient noise or sensory dropout;
- food collection, starvation, path efficiency, and collision outcomes;
- the fraction of decisions suppressed by the movement accumulator.

Use the checked-in MLP REINFORCE feature path for the first learned-policy study:
train matched seeds with and without the two curvature features while holding the
locomotor speed law fixed, then separately ablate the speed law. This distinguishes
the value of curvature as a policy observation from the value of curvature-driven
speed control.

The first simulator experiment should be interpreted as an engineering port.
A biological claim requires robustness across environments and a sensing model
that can plausibly be mapped to head sweeps or temporal chemosensation.
