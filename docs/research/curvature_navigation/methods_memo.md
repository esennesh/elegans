# Methods memo

## Odor landscape

The continuous demonstration uses a single-source scalar field

```math
C(x,y)=\exp\left[-\tfrac12(U^2+V^2)-\beta U^4-\gamma U^2V^2\right].
```

where `U,V` are translated, rotated, and anisotropically scaled coordinates
centered on the source. Unequal spatial scales make the field non-circular, and
the quartic terms make it non-Gaussian. The exponent is nonnegative and vanishes
only at the source, so the field has one unique maximum.

## Sensory input

At every step the agent samples concentration on an egocentric 3-by-3 stencil:
the center; forward, backward, left, and right points; and four diagonals. All
axial samples are one configured sensor spacing from the body center; diagonal
samples are one spacing along each body axis and therefore √2 spacings
from the center. These nine concentrations are the agent's only environmental
inputs.

Centered finite differences give the body-frame gradient `g = ∇C` and Hessian
`H = ∇²C`. This is the smallest symmetric two-dimensional stencil
that identifies both gradient components and all three independent Hessian
components.

## Field geometry

Let `s = ‖g‖`, `u = g/s`, and let `u_perp` be a 90-degree rotation of `u`. The
signed curvature of the uphill gradient streamline is

```math
\kappa_{\mathrm{flow}}
=\frac{u_\perp^\mathsf{T}Hu}{s}
=\frac{\operatorname{cross}(g,Hg)}{\lVert g\rVert^3}.
```

This is the curvature that drives speed because it measures how the direction
the agent is following bends across space. For comparison, the implementation
also logs iso-odor contour curvature,

```math
\kappa_{\mathrm{level}}
=\frac{g_y^2H_{xx}-2g_xg_yH_{xy}+g_x^2H_{yy}}
{\lVert g\rVert^3}.
```

The two are not interchangeable. In a radial Gaussian field, uphill streamlines
are straight even though its circular level sets are curved.

Curvature is undefined where the gradient vanishes. A continuous confidence
term handles that limit:

```math
q=\frac{s^2}{s^2+s_0^2}.
```

where `s_0` is the configured gradient floor. Estimates remain finite, and
weak-signal regions are treated conservatively rather than mislabeled as
low-curvature regions.

## Continuous speed response

Forward speed is

```math
v(\kappa,q)=v_{\min}+
(v_{\max}-v_{\min})
\frac{q}{1+(|\kappa|/\kappa_{1/2})^p}.
```

The formula is bounded and continuous. With reliable sensing, zero curvature
maps to `v_max`, curvature magnitude `κ_1/2` maps to the midpoint, and very high
curvature approaches `v_min`. Weak-signal confidence also moves speed toward the
conservative minimum.

## Locomotor dynamics

Steering and speed are independent:

```math
\theta_{t+1}=\theta_t+\omega_t\Delta t,
\qquad
p_{t+1}=p_t+v_t(\cos\theta_{t+1},\sin\theta_{t+1})\Delta t.
```

The controller derives angular rate `ω_t` from the normalized concentration
contrast between the forward-left and forward-right stencil samples. This is a
separate bilateral steering reflex; it does not use the curvature estimate.
Keeping angular rate independent of speed matters: slowing then permits a
tighter realized path curvature `ω_t/v_t`, rather than merely replaying the same
spatial path more slowly.

## Main-simulator port

The grid simulator evaluates its scalar multi-food concentration on the same
nine-point stencil. When `use_local_gradient_state` is enabled, the policy's food
gradient is reconstructed from that stencil in the body frame and rotated back
to world coordinates; it does not use the analytic vector computed from food
coordinates. Existing predator-vector sensing is retained because the simulator
does not yet define a scalar predator odor field.

Fractional forward speed is realized with a deterministic movement accumulator.
A speed of 0.4 cells per tick therefore produces exactly four translations over
ten eligible ticks, apart from collisions. On a speed-suppressed tick the worm
may still rotate, keeping angular response separate from translational speed.
Curvature-induced pauses are explicitly distinguished from wall collisions and
voluntary `STAY` actions, and they are exempt from stuck and anti-dithering
penalties.

`BrainParams` exposes signed streamline curvature, signed level-set curvature,
confidence, and commanded locomotion speed. The MLP REINFORCE architecture can
optionally append two of those quantities to its legacy two-feature input:

```math
z_\kappa=\tanh(\kappa_{\mathrm{flow}}/\kappa_{\mathrm{feature}}),
\qquad z_q=q.
```

The example configuration enables these two policy features, giving the network
four inputs: gradient strength, relative gradient angle, normalized signed
streamline curvature, and curvature confidence. All opt-in flags default to
false, so existing models and checkpoints retain their two-input architecture.
