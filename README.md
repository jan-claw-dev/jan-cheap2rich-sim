# Cheap2Rich Story-Aware Simulation Research

This repository kicks off the research vision Jan outlined: blend SciML-style solvers (a la Chris Rackauckas) with latent-space aligners so we can simulate and explain convergence decisions before running expensive experiments.

## Research goals

1. **Story-driven simulation dashboard** – Use deterministic solvers to model latent aligner dynamics and wrap them in a narrative layer (notes, convergence insights, next-action suggestions).
2. **SciML-informed surrogate models** – Couple DifferentialEquations.jl-style solvers with data-driven residual corrections (like DA-SHRED) so the simulator learns from contrastive SINDy runs.
3. **Experiment foresight** – Forecast how parameter tweaks or data refinements affect SSIM/latent metrics, then deliver those insights back to Jan’s personal site as short stories or warnings.

## Strategy

- Document the latent alignment pipeline from `Cheap2Rich.py`, flag the adjustable quantities (contrastive loss, gating strength, frequency attention contributions), and map them to an ODE-style latent drift.
- Build a Julia simulation script that solves such an ODE, produces a `Story` summary (status, confidence, note), and dumps JSON artifacts for the dashboard.
- Create a lightweight dashboard that simulates parameter sweeps, surfaces projected metric curves, and feeds those outputs into the personal site’s blog or research feeds.

## Julia simulation prototype

`scripts/sci_story.jl` is the first concrete step on this path. It:

- Defines a simple latent ODE that mixes drift, gating, and oscillatory perturbations.
- Solves it with `DifferentialEquations.jl` (calls `Tsit5()` with tight tolerances) and extracts a trajectory plus metadata.
- Builds a story object (status, slope, confidence, note) so the experiment forecast is immediately human-readable.
- Writes the trajectory + story payload to `artifacts/sci_story.json` so the personal site can consume it (or you can inspect it before generating plots).

### Running the Julia simulation

Ensure you have Julia installed and the dependencies available:

```bash
julia -e 'using Pkg; Pkg.add("DifferentialEquations"); Pkg.add("JSON")'
julia scripts/sci_story.jl
```

This produces `artifacts/sci_story.json` with the trajectory and story summary. The file can be picked up by a dashboard or rendered directly on the personal site as a warning/insight block.

## What’s next

- Analyze `Cheap2Rich.py` to translate the current latent alignment steps (contrastive projector, frequency attention gating, SINDy/Lasso) into the solver parameters.
- Expand the narrative layer with metrics such as SSIM forecasts, gating energy, or risk levels.
- Plan how the simulator should push artifacts (plots, JSON) back into `jan-personal-site` so the site can visualize that predictive story.
