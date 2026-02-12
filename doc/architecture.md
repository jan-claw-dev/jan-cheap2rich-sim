# Architecture sketch

## Data sources
- `Cheap2Rich.py` runs bring latent encoders, contrastive aligners, and SINDy/Lasso into the same pipeline. Each training step produces:
  - A latent vector trace (before/after gating and aligner).
  - Frequency gating masks and attention weights.
  - SSIM/decoder outputs per epoch.

## Simulation core
- The simulator will treat the latent evolution as a differential equation (drift + gating).  
- The SciML stack (DifferentialEquations.jl or a Python solver) models this latent path with controllable parameters (learning rate, gating strength, noise).  
- A surrogate residual function (e.g., poly+Lasso approximator) plugs into the solver to mirror SINDy-style corrections.

## Narrative layer
- Each simulation timestep yields a `Story` containing:
  - Metric snapshot: projected SSIM, gating energy, convergence confidence.
  - Recommendation: e.g., "Slow down gating" or "Add contrastive epochs".
  - Visual hooks: colors, icon, or simple line for the personal site carousel view.

## Dashboard pipeline
1. Run simulator with several parameter sets.  
2. Save JSON artifacts/plots to `reports/` or `artifacts/` for the portfolio site to read.  
3. Inject automated story summaries into the site’s blog or reading sections to explain what the simulation says about next experiments.
