# Curvature-navigation artifacts

These are the curated outputs of the default 20-heading curvature-navigation
demonstration:

- `curvature_taxis_demo.png`: six-panel visual diagnostic;
- `curvature_taxis_agent.mp4`: 1280-by-720 H.264 animation of the actual
  adaptive trajectory at 2x playback;
- `summary.json`: configuration, behavioral outcomes, estimator validation, and
  curvature-speed statistics;
- `heading_sweep.csv`: adaptive and matched-constant outcomes by initial
  heading.

The larger per-transition CSV files are reproducible but not duplicated here.
Generate them under the ignored `exports/` workspace with:

```bash
uv run python scripts/run_curvature_taxis_demo.py \
  --output-dir exports/curvature_navigation \
  --video
```
