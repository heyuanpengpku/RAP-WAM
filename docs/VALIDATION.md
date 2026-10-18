# Validation Scope

The release separates software contracts, launch integration and task performance. Actual results are recorded in [provenance/validation.json](../provenance/validation.json) and the [Release Audit](AUDIT_REPORT.md).

## Runtime Contracts

The CPU suite uses temporary synthetic data and reduced models. It checks both data interfaces, consecutive 16-step chunks, episode splits, training-only normalization, uint8 image scaling, language labels and context order, conversion, preflight, launch arguments, auxiliary losses, failure handling, continuation and checkpoint publication.

The public policy classes are exercised with a one-layer Wan backbone and a synthetic latent encoder. CPU attention and conditioning/normalization precision adapters allow those contracts to run without CUDA. They validate policy composition and finite gradients, not the pretrained 5B model or the external VAE.

Trainer tests use an explicit small policy to check optimizer updates, shortened loaders with gradient accumulation, saved normalization, next-epoch continuation, output-directory preservation and failed-save behavior. Nonfinite losses are rejected before an optimizer update.

```bash
python -m unittest discover -s tests -v
python scripts/check_docs.py
python -m rap_wam.check_release
```

FFmpeg/ffprobe must be installed for the video/parquet conversion test. A skipped conversion test is not complete conversion validation.

## Installation and Launch Integration

A fresh Linux/Python 3.10 environment installs the declared training-plus-data dependencies with official PyTorch 2.8.0 CPU wheels. Dependency consistency, module imports, both resolved Hydra profiles and the runtime suite are checked there. It does not inherit the earlier training environment's Python packages.

A separate two-process CPU DDP probe launches through the actual shell script, torchrun, Hydra entry, preflight and training workspace. It uses synthetic 28D data, non-model asset fixtures and a small test policy, then verifies an optimizer update and checkpoint progress. This tests local Gloo communication and launch wiring, not multi-node networking or GPU DDP.

The installed wheel is checked separately for both profiles, public metadata/import locations and all four console-entry help commands. Source ZIP and sdist contents are checked against the release manifest, including documentation, media, examples and CI configuration.

## Native CUDA Projection Heads

A separate RTX 4090 probe executes the actual video head and 14D/28D CAP head classes extracted from source, with BF16 parameters and native CUDA autocast. It checks finite forward/backward results without CPU adapters. The available isolated CUDA environment uses PyTorch 2.9.1+cu128, so this evidence is distinct from the pinned 2.8.0 CPU installation. It does not run the complete transformer, VAE, pretrained weights or training pipeline on CUDA.

## Documentation and Publication Checks

Documentation checks cover local references/anchors, figure references, heading hierarchy, display names, shell/Python block syntax and BibTeX consistency. Failure fixtures confirm those checks reject broken references and naming regressions. Desktop/mobile README previews verify loaded figures and absence of horizontal page overflow.

Publication checks inspect filenames and decoded text, reject symbolic links/unexpected binaries, verify the attributed media hashes, and scan for legacy identifiers, CJK text, private paths and credential patterns. Local model/data/venv directories and weight/environment files are excluded from release enumeration. Required upstream copyright/license notices and formal paper authors are preserved.

Hosted GitHub Actions has not run for this local release. The supplied workflow is checked locally and runs after publication when enabled.

## External Validation Still Required

No full pretrained 5B training/convergence run, full-model CUDA test, multi-node GPU communication test, simulator evaluation or physical-robot evaluation is claimed. Supply the actual demonstrations, model assets, CUDA/driver environment and deployment/evaluation integration before making new performance claims. See [Reproducibility](REPRODUCIBILITY.md) for differences from the paper.

The 0.2.2 final media review leaves runtime/test code and training configurations unchanged from 0.2.0. Its additional checks cover all 16 task/video mappings, 32 attributed assets, decoded GIF frames/durations/loops, sampled animation changes, source links, responsive layout and source-package inclusion. The 39 runtime tests remain prior 0.2.0 evidence, not a new media-release training run.

The 0.2.3 README structure review additionally checks the first-figure order, training-before-gallery navigation and collapsed/expanded layouts at desktop and mobile widths. It preserves the complete media set and carries forward the unchanged 0.2.0 runtime evidence.
