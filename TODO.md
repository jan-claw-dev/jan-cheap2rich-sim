# TODO

- [ ] Analyze `Cheap2Rich.py` to extract the latent alignment dynamics (contrastive, frequency gating, SINDy/Lasso) and map them to differential equations.
- [x] Draft `doc/architecture.md` describing how SciML solvers will feed narrative insights into the personal site.
- [x] Prototype a Julia simulation script (`scripts/sci_story.jl`) that integrates an ODE solution and outputs a narrative summary.
- [x] Define metrics for the simulated stories via the `scripts/contrastive_story.py` output (gaps, loss trace, confidence).
- [ ] Plan how the new dashboard will push data/artifacts back into the `jan-personal-site` repo.
