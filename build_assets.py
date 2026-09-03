#!/usr/bin/env python3
"""Generate every web asset for the SVDtrunc project page from the Overleaf source.

Only depends on the Python standard library plus two command line tools that
ship with macOS / poppler:

  sips      JPEG <-> BMP conversion and resampling
  pdftoppm  rasterising the vector figures

The paper's qualitative figures are single JPEGs up to 6300x10600 px (10-14 MB
each) holding a grid of images: one row per prompt, one column per compression
level. Shipping those as-is would make the page unusable, so we cut them into
per-cell JPEGs. sips can crop, but its --cropOffset is measured from the centre
and clamps unpredictably, so we do the crop ourselves: a BMP is just padded rows
of BGR bytes, which makes cropping a byte-slice and needs no image library.

Usage:  python3 build_assets.py [--source DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# figure manifest
# --------------------------------------------------------------------------

# Vector figures: (source, output stem, target width in px, dpi for rasterising)
VECTOR_FIGURES = [
    ("front_plot.pdf", "frontier", 1600, 300),
    ("method_long.pdf", "method", 2200, 300),
    ("Block_high_level.drawio.pdf", "mmdit_block", 1600, 300),
]

# Raster figures kept whole: (source, output stem, target width)
RASTER_FIGURES = [
    ("Block_Sensitivity.jpg", "block_sensitivity", 2000),
    ("cmmd_validation.jpg", "cmmd_validation", 1600),
]

# Grids to slice: (source, stem, n_cols, n_rows, cell width, jpeg quality)
GRIDS = [
    ("tmp_front_examples.jpg", "hero", 5, 2, 640, 82),
    ("final_add_qualitative_1.jpg", "qual1", 6, 10, 512, 78),
    ("final_add_qualitative_2.jpg", "qual2", 6, 10, 512, 78),
    ("final_domain_drift.jpg", "drift", 6, 10, 512, 78),
    ("final_memory_loss.jpg", "memory", 6, 10, 512, 78),
]


# --------------------------------------------------------------------------
# shell helpers
# --------------------------------------------------------------------------

def run(*args: object) -> str:
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{args[0]} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def need(tool: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(f"error: required tool '{tool}' not found on PATH")


# --------------------------------------------------------------------------
# minimal BMP reader/writer (top-down rows of unpadded BGR bytes)
# --------------------------------------------------------------------------

class Bmp:
    def __init__(self, rows: list[bytes], w: int, h: int, nbytes: int):
        self.rows, self.w, self.h, self.nbytes = rows, w, h, nbytes

    @classmethod
    def read(cls, path: Path) -> "Bmp":
        raw = path.read_bytes()
        off = struct.unpack_from("<I", raw, 10)[0]
        w, h = struct.unpack_from("<ii", raw, 18)
        bpp = struct.unpack_from("<H", raw, 28)[0]
        if bpp not in (24, 32):
            raise RuntimeError(f"unsupported bit depth {bpp}")
        nb = bpp // 8
        stride = ((w * nb + 3) // 4) * 4
        bottom_up, h = h > 0, abs(h)
        rows = []
        for y in range(h):
            src = (h - 1 - y) if bottom_up else y
            base = off + src * stride
            rows.append(raw[base:base + w * nb])
        return cls(rows, w, h, nb)

    def crop(self, x0: int, y0: int, x1: int, y1: int) -> "Bmp":
        nb = self.nbytes
        return Bmp([r[x0 * nb:x1 * nb] for r in self.rows[y0:y1]],
                   x1 - x0, y1 - y0, nb)

    def write(self, path: Path) -> None:
        nb, w, h = self.nbytes, self.w, self.h
        stride = ((w * nb + 3) // 4) * 4
        pad = b"\x00" * (stride - w * nb)
        body = b"".join(self.rows[y] + pad for y in range(h - 1, -1, -1))
        hdr = struct.pack("<2sIHHI", b"BM", 14 + 40 + len(body), 0, 0, 14 + 40)
        dib = struct.pack("<IiiHHIIiiII", 40, w, h, 1, nb * 8, 0, len(body),
                          2835, 2835, 0, 0)
        path.write_bytes(hdr + dib + body)

    # gutter detection

    def _pale(self, x: int, y: int, thr: int) -> bool:
        o = x * self.nbytes
        r = self.rows[y]
        return r[o] >= thr and r[o + 1] >= thr and r[o + 2] >= thr

    def pale_cols(self, thr: int = 243, step: int = 7) -> list[float]:
        ys = range(0, self.h, step)
        n = len(ys)
        return [sum(1 for y in ys if self._pale(x, y, thr)) / n
                for x in range(self.w)]

    def pale_rows(self, thr: int = 243, step: int = 7) -> list[float]:
        xs = range(0, self.w, step)
        n = len(xs)
        return [sum(1 for x in xs if self._pale(x, y, thr)) / n
                for y in range(self.h)]


def gutters(prof: list[float], min_frac: float = 0.985) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for i, f in enumerate(prof):
        if f >= min_frac and start is None:
            start = i
        elif f < min_frac and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(prof)))
    return out


def bands(prof: list[float], n_expected: int) -> list[tuple[int, int]]:
    """Content bands lying between the white gutters.

    The label strip across the top ("FLUX.dev  80%  68% ...") is white with dark
    text, so it reads as its own short band rather than as part of a gutter. Any
    band far shorter than the median is dropped, which removes it.
    """
    n = len(prof)
    spans: list[tuple[int, int]] = []
    cur = 0
    for a, b in gutters(prof):
        if a > cur:
            spans.append((cur, a))
        cur = b
    if cur < n:
        spans.append((cur, n))
    if not spans:
        raise RuntimeError("no content bands found")
    med = sorted(b - a for a, b in spans)[len(spans) // 2]
    kept = [(a, b) for a, b in spans if (b - a) > med * 0.5]
    if len(kept) != n_expected:
        raise RuntimeError(f"expected {n_expected} bands, found {len(kept)}: {kept}")
    return kept


# --------------------------------------------------------------------------
# build steps
# --------------------------------------------------------------------------

def build_vector(src_dir: Path, out_dir: Path, tmp: Path) -> None:
    for name, stem, width, dpi in VECTOR_FIGURES:
        src = src_dir / name
        if not src.exists():
            print(f"  skip {name} (missing)")
            continue
        stage = tmp / f"{stem}_raw"
        run("pdftoppm", "-r", dpi, "-png", "-singlefile", src, stage)
        png = stage.with_suffix(".png")
        run("sips", "-s", "format", "png", "--resampleWidth", width,
            png, "--out", out_dir / f"{stem}.png")
        png.unlink(missing_ok=True)
        print(f"  {stem}.png")


def build_raster(src_dir: Path, out_dir: Path) -> None:
    for name, stem, width in RASTER_FIGURES:
        src = src_dir / name
        if not src.exists():
            print(f"  skip {name} (missing)")
            continue
        run("sips", "-s", "format", "jpeg", "-s", "formatOptions", 84,
            "--resampleWidth", width, src, "--out", out_dir / f"{stem}.jpg")
        print(f"  {stem}.jpg")


def build_grid(src: Path, stem: str, n_cols: int, n_rows: int, out_dir: Path,
               tmp: Path, cell_w: int, quality: int, thumb_w: int = 224,
               inset: int = 2) -> None:
    work_w = n_cols * (cell_w + 96)
    work = tmp / "_grid.bmp"
    cellbmp = tmp / "_cell.bmp"
    run("sips", "-s", "format", "bmp", "--resampleWidth", work_w,
        src, "--out", work)
    img = Bmp.read(work)
    col_bands = bands(img.pale_cols(), n_cols)
    row_bands = bands(img.pale_rows(), n_rows)

    for r, (y0, y1) in enumerate(row_bands):
        for c, (x0, x1) in enumerate(col_bands):
            img.crop(x0 + inset, y0 + inset, x1 - inset, y1 - inset).write(cellbmp)
            run("sips", "-s", "format", "jpeg", "-s", "formatOptions", quality,
                "--resampleWidth", cell_w, cellbmp,
                "--out", out_dir / f"{stem}_r{r:02d}_c{c}.jpg")
        x0, x1 = col_bands[0]
        img.crop(x0 + inset, y0 + inset, x1 - inset, y1 - inset).write(cellbmp)
        run("sips", "-s", "format", "jpeg", "-s", "formatOptions", 70,
            "--resampleWidth", thumb_w, cellbmp,
            "--out", out_dir / f"{stem}_r{r:02d}_thumb.jpg")

    work.unlink(missing_ok=True)
    cellbmp.unlink(missing_ok=True)
    print(f"  {stem}: {n_cols}x{n_rows} = {n_cols * n_rows} cells + {n_rows} thumbs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="../Upload_Camera_Ready___Importance_Aware_"
                                        "Low_Rank_Distillation_of_Diffusion_Transformers",
                    help="Overleaf project directory")
    ap.add_argument("--out", default="assets", help="asset output directory")
    args = ap.parse_args()

    need("sips")
    need("pdftoppm")

    here = Path(__file__).resolve().parent
    src_dir = (here / args.source).resolve() / "images"
    if not src_dir.is_dir():
        sys.exit(f"error: no images directory at {src_dir}")

    out = (here / args.out).resolve()
    fig_dir, gal_dir = out / "figures", out / "gallery"
    tmp = out / ".tmp"
    for d in (fig_dir, gal_dir, tmp):
        d.mkdir(parents=True, exist_ok=True)

    print("vector figures")
    build_vector(src_dir, fig_dir, tmp)
    print("raster figures")
    build_raster(src_dir, fig_dir)
    print("image grids")
    for name, stem, ncol, nrow, cw, q in GRIDS:
        src = src_dir / name
        if not src.exists():
            print(f"  skip {name} (missing)")
            continue
        build_grid(src, stem, ncol, nrow, gal_dir, tmp, cw, q)

    shutil.rmtree(tmp, ignore_errors=True)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\ndone: {total / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    main()
