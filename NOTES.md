# Maintenance notes

Internal notes for the SVDtrunc project page. The public-facing description lives in
[README.md](README.md). Delete this file before publishing if you would rather it were not
visible.

Static project page for **Importance-Aware Low-Rank Distillation of Diffusion Transformers**
(Zavadski, Heid, Kalšan, Roth, Rother).

No build step, no dependencies, no framework: one `index.html` plus an `assets/` folder.
Open `index.html` in a browser to view it locally.

```
index.html          the whole page: markup, CSS and JS in one file
assets/figures/     5 figures rasterised or downscaled from the paper
assets/gallery/     250 per-image cells + 42 thumbnails cut from the paper's grids
build_assets.py     regenerates everything in assets/ from the Overleaf source
```

## Before publishing: 3 things to fill in

**1. Links.** Near the top of the `<script>` block at the bottom of `index.html`:

```js
const CONFIG = {
  arxiv:   "",   // e.g. "https://arxiv.org/abs/2603.01234"
  code:    "",   // e.g. "https://github.com/<you>/svdtrunc"
  weights: "",   // e.g. "https://huggingface.co/<you>/svdtrunc"
};
```

An empty string renders that button as a greyed-out "soon" chip, so the page is safe to
publish before the code and weights are live. Fill each one in as it goes up. No other edit
is needed. The paper and supplement are reached through the arXiv link rather than getting
buttons of their own.

**2. The venue badge.** It currently reads `ECCV 2026`, inferred from `eccv.sty` and the
camera-ready source. If that is wrong or not yet public, edit the `<span class="venue">`
in the header and the `booktitle`/`year` in the BibTeX block.

**3. The BibTeX entry.** Written by hand since the paper is not on arXiv yet. Replace it
with the official entry once you have one.

## Deploying to GitHub Pages

The page needs a URL before you can put it in the arXiv submission, so publish it first. The
page lives on a `gh-pages` branch so that `main` stays free for the code release:

```bash
git init -b gh-pages
git add . && git commit -m "Project page"
git remote add origin git@github.com:vislearn/SVDtrunc.git
git push -u origin gh-pages
```

Then **Settings → Pages → Source: Deploy from a branch → `gh-pages` / `(root)`**. The page
appears at `https://vislearn.github.io/SVDtrunc/` after a minute or two. `.nojekyll` is already
present, so GitHub serves the files as-is without running Jekyll over them.

When the code is ready, push it to `main` in the same repository and make `main` the default
branch. The page keeps building from `gh-pages` and the URL does not change.

`assets/` is about 19 MB, which is fine for Pages (1 GB soft limit) and loads lazily in the
browser, so only the teaser row is fetched up front.

## Regenerating the assets

```bash
python3 build_assets.py                       # expects the Overleaf dir alongside this one
python3 build_assets.py --source /path/to/overleaf --out assets
```

Requires only `sips` (ships with macOS) and `pdftoppm` (`brew install poppler`). No Python
packages. On Linux, replace the `sips` calls with ImageMagick equivalents.

The interesting part is the grid slicing. The paper's qualitative figures are single JPEGs up
to 6300×10600 px (10 to 14 MB each) holding one row per prompt and one column per compression
level. Shipping those as-is would make the page unusable, so `build_assets.py` locates the
white gutters between cells, cuts each cell out, and writes it as a small JPEG. Cropping is
done by byte-slicing BMP rows, which avoids needing Pillow or ImageMagick. Note that
`sips --cropOffset` measures from the image centre and clamps unpredictably, so it is not
usable for this.

If a figure is ever re-exported with a different number of rows or columns, update the
`GRIDS` table at the top of the script. The band detector raises an error rather than
silently misaligning if the counts do not match what it finds.

## Notes on the content

- Every number in the four results tables is transcribed from `tables/*.tex`. Bold marks the
  best value in a group, following the paper.
- The training-free section states plainly that block pruning is still marginally ahead at a
  90% budget, and that the ordering inverts from 80% downward. This is deliberate, since the
  crossover is a more credible framing than showing only the columns where the gap is large.
- The comparison section's 40 scene captions were written from the images themselves. The real
  prompts were JoyCaption-generated and are not in the source, so reword them freely. They
  live in the `SETS` array.
- Missing values in the tables are marked `n/a` rather than with a dash.
- Equations are rendered by MathJax from a CDN. Without a network connection they fall back
  to displaying raw LaTeX. Vendor `tex-chtml.js` into `assets/` if you want it fully offline.
- The page was validated for markup balance, JS syntax and asset-path resolution, but it has
  **not** been rendered in a browser, because headless Chrome cannot start in the environment
  it was built in. Please give it a visual once-over, dark mode included.
