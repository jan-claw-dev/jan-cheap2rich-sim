# Cheap2Rich Story-Aware Simulation Research

This repository kicks off the research vision Jan outlined: blend SciML-style solvers (a la Chris Rackauckas) with latent-space aligners so we can simulate and explain convergence decisions before running expensive experiments.

## Research goals

1. **Story-driven simulation dashboard** – Use deterministic ODE/PDE solvers to model latent aligner dynamics and wrap them in a narrative layer (notes, convergence insights, next-action suggestions).
2. **SciML-informed surrogate models** – Couple DifferentialEquations.jl-style solvers or components with data-driven residual corrections (like DA-SHRED) so the simulator learns from Cheap2Rich’s contrastive/SINDy runs.
3. **Experiment foresight** – Forecast how parameter tweaks or data refinements affect SSIM/latent metrics, then present the implications as short narratives for Jan’s site.

## Strategy

- Start by documenting the latent alignment pipeline from `Cheap2Rich.py` and identify the parts that can map to ODEs (e.g., latent drift, gating dynamics, frequency attention).  
- Prototype a small Julia/Python hybrid where SciML solvers estimate how a latent evolves under frequency gating; wrap those outputs in a ``Story" object (status + note).  
- Build a lightweight dashboard that simulates sequences of runs and highlights where GAN-based decoders might fail, so the site can show a simulated caution before you commit to retraining fully.

## What’s next

- Sketch the core data flow (latent state → sim → story) in `doc/architecture.md`.  
- Seed `scripts/initial_sim.py` (or a Julia notebook) with a toy differential equation mimicking latent drift plus a placeholder narrative generator.
- Track outstanding questions in `TODO.md` and link to Jan’s personal site for eventual integration.
