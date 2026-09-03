# Importance-Aware Low-Rank Distillation of Diffusion Transformers

**Denis Zavadski**\*, **Sebastian Heid**\*, Damjan Kalšan, Stefan Roth, Carsten Rother
<br><sub>\*Equal contribution. Heidelberg University · Zuse School ELIZA · TU Darmstadt · hessian.AI</sub>

### [Project page](https://vislearn.github.io/SVDtrunc/) · arXiv (soon) · Code (soon) · Weights (soon)

This repository hosts the project page for **SVDtrunc**, an importance-aware truncated SVD
compression scheme for Diffusion Transformers.

Truncated SVD is known to fail on large language models, where naive low-rank approximation of
weight matrices can collapse the model outright. We find that Diffusion Transformers behave in
the opposite way: quality degrades smoothly as rank is removed, and the exploitable redundancy
is spread across projection matrices throughout the whole network rather than concentrated in a
few blocks. SVDtrunc exploits this by ranking blocks through compression probing, allocating
rank non-uniformly under a global parameter budget, and fine-tuning with modular knowledge
distillation and a rectified-flow objective.

Applied to FLUX.dev, it retains near-full quality at 68% of the original parameter count
(0.75% mean reduction across GenEval, HPSv2 and DPG) and stays competitive at 57%.

## Status

| | |
|---|---|
| Project page | this repository |
| Code | released shortly |
| Weights | on Hugging Face |

## Citation

```bibtex
@inproceedings{zavadski2026svdtrunc,
  title     = {Importance-Aware Low-Rank Distillation of Diffusion Transformers},
  author    = {Zavadski, Denis and Heid, Sebastian and Kal{\v{s}}an, Damjan
               and Roth, Stefan and Rother, Carsten},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## About this repository

A static page: one `index.html` with no build step or dependencies, plus an `assets/` folder.
`build_assets.py` regenerates the assets from the paper's LaTeX source. See
[NOTES.md](NOTES.md) for maintenance details.
