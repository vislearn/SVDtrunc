# SVDtrunc: Importance-Aware Low-Rank Distillation of Diffusion Transformers

[**Denis Zavadski**](https://sipirius.github.io/)<sup>1,2,\*</sup>, **Sebastian Heid**<sup>1,2,\*</sup>, Damjan Kalšan<sup>1</sup>, Stefan Roth<sup>2,3,4</sup>, Carsten Rother<sup>1,2</sup>

<sub><sup>1</sup>Heidelberg University · <sup>2</sup>Zuse School ELIZA · <sup>3</sup>TU Darmstadt · <sup>4</sup>hessian.AI · <sup>\*</sup>Equal contribution</sub>

### [Project page](https://vislearn.github.io/SVDtrunc/) · [arXiv](https://arxiv.org/abs/2609.04646) · Weights (soon)

> [!NOTE]
> **Code release coming soon.** Training, compression and evaluation code will land in this
> repository shortly, with checkpoints on Hugging Face. In the meantime, the
> [project page](https://vislearn.github.io/SVDtrunc/) has qualitative comparisons across all
> compression levels, the full results tables, and a walkthrough of the method.

## Abstract

Diffusion Transformers (DiTs) have emerged as a dominant architecture for high-quality
text-to-image generation, yet their scale poses challenges for efficient deployment. While
truncated singular value decomposition (SVD) is a principled tool for parameter reduction,
evidence from large language models (LLMs) suggests that naive low-rank approximation can cause
catastrophic failure. In contrast, we find that truncated SVD in DiTs produces smooth degradation
even under substantial global compression, with redundancy distributed across projection matrices
throughout the whole network rather than concentrated in a few transformer blocks.

Building on these insights, we introduce *SVDtrunc*, a two-step block-level compression scheme,
first allocating ranks across blocks and compressing the least important ones via truncated SVD
under a global parameter budget, and then fine-tuning all blocks with modular knowledge
distillation and a rectified-flow objective.

We apply SVDtrunc to FLUX.dev across compression levels ranging from 40% to 90% of the original
parameter count. Across three benchmarks, GenEval, HPSv2, and DPG, we outperform all competing
approaches. Notably, and in contrast to prior work, we retain near-full performance at 68% and
remain competitive even at 57% of the original parameter budget. Furthermore, we show that
SVDtrunc complements step distillation and achieves strong results even without fine-tuning,
positioning it as a practical continuation of efficiency improvements beyond diffusion step
reduction for large-scale generative models.

## Results

Compression of FLUX.dev, where *R* is the mean quality reduction across GenEval, HPSv2 and DPG
relative to the uncompressed model.

| Method | Params ↓ | GenEval ↑ | HPSv2 ↑ | DPG ↑ | *R* ↓ |
|---|---|---|---|---|---|
| FLUX.dev | 100% | 0.647 | 31.70 | 83.9 | 0.00 |
| PPCL (CVPR '26) | 68% | 0.605 | n/a | 80.0 | 5.55 |
| **SVDtrunc-m** | **68%** | **0.645** | 31.27 | **83.4** | **0.75** |
| **SVDtrunc-s** | **57%** | 0.616 | 31.22 | 82.6 | **2.60** |

See the [project page](https://vislearn.github.io/SVDtrunc/) for the full comparison, the
training-free setting, and results on step-distilled FLUX.schnell.

## Release checklist

- [x] Project page
- [x] arXiv preprint
- [ ] Inference code
- [ ] Compressed checkpoints on Hugging Face
- [ ] Compression and probing code
- [ ] Training and distillation code

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
