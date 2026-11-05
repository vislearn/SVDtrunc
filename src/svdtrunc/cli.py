"""Command line for SVDtrunc: compression of FLUX.1 and image generation with compressed models.
`svdtrunc <command> --help` lists a command's options.

Checkpoints are local files or Hub files written `<owner>/<repo>/<file>`. Whether a checkpoint derives from
FLUX.1-dev or FLUX.1-schnell is detected; text encoders, autoencoder and sampling defaults follow it.

Generate images:

    svdtrunc generate --checkpoint CHECKPOINT --prompt "a red fox in the snow" --output-dir outputs

Compress FLUX.1 (dev or schnell, matching the model) to the structure of a released model, without fine-tuning:

    svdtrunc compress --like CHECKPOINT --output trimmed.safetensors

One round of importance-aware compression (Sec. 3 of the paper):

    svdtrunc render   --checkpoint current.safetensors --prompts prompts.json --output-dir ref
    svdtrunc render   --checkpoint current.safetensors --prompts prompts.json \\
                      --probe-block double_blocks.13 --output-dir probes/double_blocks.13   # one job per block
    svdtrunc rank     --checkpoint current.safetensors --reference ref --probes probes --output ranking.json
    svdtrunc schedule --checkpoint current.safetensors --ranking ranking.json --output next.json
    svdtrunc compress --checkpoint current.safetensors --spec next.json --output next.safetensors

In the first round, `current.safetensors` is the base model, e.g. `black-forest-labs/FLUX.1-dev/flux1-dev.safetensors`.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from torch import nn

from svdtrunc.cmmd import CLIP_IMAGE_MODEL, ClipImageEmbedder, mmd
from svdtrunc.compression import block_paths, compress, describe, parameters_removed, step_block_spec
from svdtrunc.flux.model import Flux
from svdtrunc.loading import (
    DEV,
    detect_base,
    load_ae,
    load_flux,
    load_structure,
    load_text_encoders,
    read_spec,
    save_flux,
)
from svdtrunc.render import encode_prompts, render, save_images
from svdtrunc.selection import PROBE_STEP, probe_spec, schedule_spec, score
from svdtrunc.spec import CompressionSpec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="svdtrunc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    command = commands.add_parser("generate", help="generate images from text prompts")
    command.add_argument("--checkpoint", required=True)
    prompts = command.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", action="append", help="a prompt; repeat for several")
    prompts.add_argument("--prompts", type=Path, help="JSON list of prompts, or a text file with one per line")
    command.add_argument("--output-dir", type=Path, default=Path("outputs"))
    command.add_argument("--seed", type=int, default=0, help="seed of the first image; image i uses seed + i")
    _add_sampling_arguments(command, device, batch_size=1)
    command.set_defaults(run=_generate)

    command = commands.add_parser("describe", help="print a checkpoint's parameters, block by block")
    command.add_argument("checkpoint")
    command.set_defaults(run=_describe)

    command = commands.add_parser("compress", help="bring a checkpoint to a target structure, truncating via SVD")
    command.add_argument(
        "--checkpoint", help="checkpoint to compress (default: the base model of --like, else FLUX.1-dev)"
    )
    target = command.add_mutually_exclusive_group(required=True)
    target.add_argument("--spec", type=Path, help="spec file, e.g. from `schedule`")
    target.add_argument("--like", help="checkpoint whose structure to copy, e.g. a released model")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--device", default=device)
    command.set_defaults(run=_compress)

    command = commands.add_parser("render", help="render prompts for block probing, optionally with one block probed")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--prompts", type=Path, required=True, help="JSON list of prompts, or one per line")
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--probe-block", help="block to compress a further --probe-step first, e.g. single_blocks.4")
    command.add_argument("--probe-step", type=float, default=PROBE_STEP)
    command.add_argument("--seed", type=int, default=42, help="seed of every image")
    _add_sampling_arguments(command, device, batch_size=4)
    command.set_defaults(run=_render)

    command = commands.add_parser("rank", help="score probes by CMMD per million parameters removed")
    command.add_argument("--checkpoint", required=True, help="checkpoint the probes were rendered from")
    command.add_argument("--reference", type=Path, required=True, help="images of the unprobed model")
    command.add_argument(
        "--probes", type=Path, required=True, help="one image directory per probed block, named after it"
    )
    command.add_argument("--probe-step", type=float, default=PROBE_STEP)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--clip-model", default=CLIP_IMAGE_MODEL)
    command.add_argument("--device", default=device)
    command.set_defaults(run=_rank)

    command = commands.add_parser("schedule", help="write the spec of the next round from a ranking")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--ranking", type=Path, required=True)
    command.add_argument("--blocks", type=int, default=30, help="number of best-ranked blocks to compress")
    command.add_argument("--start", type=float, default=0.25, help="step of the best-ranked block")
    command.add_argument("--target", type=float, default=1.19e9, help="parameters to remove in this round")
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(run=_schedule)

    args = parser.parse_args(argv)
    args.run(args)


# Private implementation


def _add_sampling_arguments(command: argparse.ArgumentParser, device: str, batch_size: int) -> None:
    command.add_argument("--ae", help="autoencoder (default: the base model's)")
    command.add_argument("--height", type=int, default=1024)
    command.add_argument("--width", type=int, default=1024)
    command.add_argument("--steps", type=int, help="denoising steps (default: 50 for FLUX.1-dev, 4 for schnell)")
    command.add_argument("--guidance", type=float, help="guidance (default: 3.5; FLUX.1-schnell has none)")
    command.add_argument("--batch-size", type=int, default=batch_size)
    command.add_argument("--device", default=device)


def _generate(args: argparse.Namespace) -> None:
    prompts = args.prompt or _read_prompts(args.prompts)
    seeds = [args.seed + index for index in range(len(prompts))]
    _render_prompts(args, prompts, seeds, probe_block=None)


def _describe(args: argparse.Namespace) -> None:
    model = load_structure(args.checkpoint)
    with torch.device("meta"):
        dense = Flux(model.params)
    removed = describe(model).removed_blocks
    print(f"{'block':<18} {'params (M)':>11} {'of dense':>9}")
    for block in block_paths(model):
        parameters = _count(model.get_submodule(block))
        share = "removed" if block in removed else f"{100 * parameters / _count(dense.get_submodule(block)):.1f}%"
        print(f"{block:<18} {parameters / 1e6:11.1f} {share:>9}")
    print(f"{'total':<18} {_count(model) / 1e6:11.1f} {100 * _count(model) / _count(dense):8.1f}%")


def _compress(args: argparse.Namespace) -> None:
    spec = read_spec(args.like) if args.like else CompressionSpec.from_json(args.spec.read_text())
    checkpoint = args.checkpoint or (detect_base(args.like) if args.like else DEV).checkpoint
    model = load_flux(checkpoint, device=args.device)
    compress(model, spec)
    save_flux(model, args.output)


def _render(args: argparse.Namespace) -> None:
    prompts = _read_prompts(args.prompts)
    _render_prompts(args, prompts, [args.seed] * len(prompts), probe_block=args.probe_block)


def _render_prompts(args: argparse.Namespace, prompts: list[str], seeds: list[int], probe_block: str | None) -> None:
    """Writes `<output-dir>/<index>.png` per prompt, skipping batches already written."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / f"{index:04d}.png" for index in range(len(prompts))]
    todo = [index for index, path in enumerate(paths) if not path.exists()]
    if not todo:
        return
    base = detect_base(args.checkpoint)
    # The text encoders are freed before the transformer is loaded, so that FLUX.1-dev fits on a 40 GB GPU.
    t5, clip = load_text_encoders(base, device=args.device)
    txt, vec = encode_prompts(t5, clip, [prompts[index] for index in todo])
    del t5, clip
    gc.collect()
    torch.cuda.empty_cache()
    model = load_flux(args.checkpoint, device=args.device)
    if probe_block:
        compress(model, probe_spec(model, probe_block, args.probe_step))
    ae = load_ae(args.ae or base.ae, device=args.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start : start + args.batch_size]
        images = render(
            model,
            ae,
            txt[start : start + len(batch)],
            vec[start : start + len(batch)],
            [seeds[index] for index in batch],
            args.height,
            args.width,
            args.steps or base.steps,
            base.guidance if args.guidance is None else args.guidance,
            base.shift,
        )
        save_images(images, [paths[index] for index in batch])
        print(f"Wrote {', '.join(str(paths[index]) for index in batch)}", flush=True)
    if torch.cuda.is_available():
        print(f"Peak GPU memory while sampling: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def _rank(args: argparse.Namespace) -> None:
    model = load_structure(args.checkpoint)
    embedder = ClipImageEmbedder(args.clip_model, device=args.device)
    reference_paths = _pngs(args.reference)
    reference = embedder.embed_files(reference_paths)
    results = []
    for directory in sorted(path for path in args.probes.iterdir() if path.is_dir()):
        paths = _pngs(directory)
        if len(paths) != len(reference_paths):
            raise ValueError(f"{directory} has {len(paths)} images, the reference has {len(reference_paths)}.")
        value = mmd(reference, embedder.embed_files(paths))
        removed = parameters_removed(model, step_block_spec(model, directory.name, args.probe_step))
        results.append(
            {"block": directory.name, "cmmd": value, "parameters_removed": removed, "score": score(value, removed)}
        )
    results.sort(key=lambda result: result["score"])
    args.output.write_text(json.dumps(results, indent=1))
    for result in results:
        block, cmmd, removed_millions = result["block"], result["cmmd"], result["parameters_removed"] / 1e6
        print(f"{block:<18} CMMD {cmmd:.4f}  removed {removed_millions:6.1f}M  score {result['score']:.3g}")


def _schedule(args: argparse.Namespace) -> None:
    model = load_structure(args.checkpoint)
    ranking = [result["block"] for result in json.loads(args.ranking.read_text())]
    spec = schedule_spec(model, ranking, args.blocks, args.start, args.target)
    args.output.write_text(spec.to_json())
    removed = parameters_removed(model, spec.linears)
    print(f"Removes {removed / 1e9:.3f}B parameters (target {args.target / 1e9:.3f}B): {args.output}")


def _read_prompts(path: Path) -> list[str]:
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    return [line.strip() for line in text.splitlines() if line.strip()]


def _pngs(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images in {directory}.")
    return paths


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


if __name__ == "__main__":
    main()
