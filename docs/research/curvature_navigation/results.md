# Results and interpretation

The default 20-heading run was regenerated after the implementation and is
preserved in `artifacts/curvature_navigation/`.

## Verified result

The adaptive agent reached the target in 36.96 simulated seconds. Its distance
to the source fell from 7.382 to 0.448 units while experienced concentration rose
from 0.0741 to 0.9930. The no-steering control exited the arena without reaching
the target.

The nine-sample curvature estimate closely tracked the analytic-derivative
reference evaluated with the same gradient-floor regularization:

- mean absolute error: 0.00496 inverse units;
- root-mean-square error: 0.00569 inverse units;
- estimated/reference correlation: 0.999997;
- sensed streamline-curvature range: -0.608 to 3.212 inverse units.

The correlation is a trajectory diagnostic, not an independent-sample
statistic: adjacent time points are strongly autocorrelated.

Forward speed covered almost the full configured range, from 0.1015 to 0.4200
units per second. Mean speed was 0.2956 in the lowest-curvature quartile and
0.1319 in the highest-curvature quartile, a difference of 0.1636. The ordinary
correlation between curvature magnitude and speed was -0.589. It is not expected
to equal -1 because the prespecified relationship is nonlinear and saturates near
the minimum speed.

Across 20 evenly spaced initial headings, both the adaptive controller and its
matched-mean constant-speed control reached the target in all 20 cases. This is
an important negative result: the run proves curvature detection and continuous
speed modulation, but it does **not** show a success-rate advantage over constant
speed in this favorable field. Harder fields, angular-rate limits, noise, and
matched energy budgets belong in the next behavioral study.

The checked-in grid-simulator configuration is an integration example, not a
trained result. A deterministic seed-42 smoke run completed normally through the
public CLI, collected 2 of 12 foods, and then starved after 280 steps. That outcome
is expected from a newly initialized REINFORCE policy and should not be read as a
test of the curvature hypothesis; the relevant result is that locally sensed
gradient and curvature features propagate through the full simulator and remain
trainable.

The primary checks are:

1. the traversed odor-field streamline curvature varies materially;
2. finite-difference estimates track the analytic reference where confidence is
   high;
3. observed speed follows the prespecified continuous inverse-curvature law;
4. the agent reaches the source using only local concentration samples; and
5. the opt-in grid integration uses the same stencil for food-gradient steering,
   exposes curvature and confidence to the policy, and modulates locomotion
   without changing default simulator behavior.

In the diagnostic figure, read the panels in this order:

- **Field and trajectory:** non-elliptic contour shape establishes that the
  landscape is not the old circular Gaussian. Color along the path represents
  speed; slower segments should coincide with visibly bending gradient flow.
- **Estimated versus reference curvature:** overlapping curves validate the
  internal estimate. Divergence near the source is expected when gradient
  confidence falls, and should not be mistaken for a zero-curvature estimate.
- **Speed response:** the black curve is the full-confidence upper envelope.
  Experienced points follow the same law and may lie below it when weak-gradient
  confidence deliberately pulls speed toward the conservative minimum.
- **Behavior:** decreasing distance and increasing concentration establish
  successful chemotaxis, while the constant-speed comparison isolates the
  effect of speed modulation from steering.
