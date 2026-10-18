# RAP-WAM Release Audit

Project: RAP-WAM  
Package: `rap_wam`  
Version: `0.2.3`  
Date: 2026-10-18

## Review Coverage

The review follows the repository from installation and source packaging through data conversion, context preparation, preflight, resolved configurations, launch scripts, normalization, policy calls, training, checkpoint continuation and documentation. Display names use RAP-WAM; executable identifiers use `rap_wam`. Publication checks preserve scholarly/upstream attribution while rejecting private infrastructure and legacy identifiers.

## Findings and Corrections

| Finding | Correction | Evidence |
|---|---|---|
| Normalization included held-out episodes despite documentation | Fit statistics only on complete episodes represented in the training split; reuse them for validation | Disjoint extreme-valued train/validation fixture |
| Dark uint8 images with maximum value 1 were treated as normalized floats | Always scale uint8 pixels by 255; preserve normalized float images | Pixel-intensity regression |
| Missing language labels silently selected empty text | Require one valid integer task index per sample for training and sampling | Explicit-label policy tests |
| Optional visual diagnostics dropped task labels | Forward labels into sampling observations | Workspace sampling regression |
| Separate asset resolvers could choose different transformer configs | Share one directory resolver across preflight and model/VAE construction | Conflicting-directory fixture |
| Asset preflight accepted incomplete weight downloads | Check nonempty VAE/weights and every referenced index shard | Missing-shard fixture |
| Vocabularies and fractional episode labels were permissive | Centralize ordered-list validation and reject invalid episode/label metadata | Invalid-vocabulary/label fixtures |
| Shard metadata could hide truncated modality arrays | Check exclusive integer boundaries, array lengths/dtypes and cross-shard shapes | Truncated-array regression |
| Dry-run did not validate Hydra overrides/modes | Compose the selected profile and validate supported options before launch | Both-profile launch/option checks |
| Output overrides could disagree with the launch record | Require RUN_DIR, normalize its path and atomically reserve a new directory | Launch argument regressions |
| A shortened epoch could leave accumulated gradients uncommitted | Bound actual batch-sampler length and calculate scheduler budget from local update counts | Short-loader optimizer/resume regression and CPU DDP probe |
| Missing resume checkpoints could start fresh training | Reject absent paths before model initialization | Failure-before-dataset test |
| Legacy checkpoint metadata could restore the old output directory | Preserve the current output destination | Legacy-payload regression |
| Failed writes could truncate the previous checkpoint/context | Publish via temporary files with atomic replacement or exclusive linking | Simulated disk failure and overwrite tests |
| Epoch-end continuation could repeat a completed epoch | Save the next epoch in new format-version-2 checkpoints | Continue into a separate output directory |
| Auxiliary schedules/readiness were absent from model state | Save extra state; load legacy state dictionaries with documented fallback | Model-state roundtrip and legacy fixture |
| Epoch summaries averaged cumulative counters | Retain the last cumulative value | Inspected log aggregation; counters covered by policy tests |
| Public checkpoint specialization bypassed sampled monitor metrics | Preserve measured metrics and defer selection until available | Public workspace monitor regression |
| Nonfinite losses could reach the optimizer | Stop before backward/update | Nonfinite-loss regression |
| Release tooling could admit links or arbitrary binaries | Reject them, validate known media, exclude runtime assets/weights/environment files | Publication fixtures and archive scans |

## Verification

The current suite contains 39 runtime tests. Final pass/fail/skip counts and the tested environment are recorded in [validation.json](../provenance/validation.json), which is refreshed after running the suite. The release also checks:

- A fresh Linux/Python 3.10 CPU dependency installation and package compatibility.
- Both Hydra profiles and all project module/target imports.
- Two-process CPU DDP through the public launch script with synthetic data and a small policy.
- Native CUDA forward/backward of the video and 14D/28D CAP heads on an RTX 4090, separately under PyTorch 2.9.1+cu128.
- Installed-wheel imports/metadata/console commands and source/sdist/ZIP file hashes.
- Documentation references, code-block syntax, naming fixtures and desktop/mobile rendering.
- Filename/text/privacy scans, media attribution/hash checks and retained upstream notices.

## Reproduction Boundary

Training-only normalization and corrected image/continuation behavior can change new runs relative to prior releases. Existing checkpoints retain their saved normalization. Legacy checkpoints cannot reconstruct missing schedule/RNG/dataloader state. The public recipes still differ from the paper in batch, split, caching and checkpoint selection.

Full 5B convergence, a complete CUDA training run, multi-node GPU networking and robot/simulator performance remain unverified. The CPU tests, DDP test policy and isolated CUDA heads do not establish those results. Hosted GitHub Actions also requires a real hosted run after publication. Detailed scope is in [VALIDATION.md](VALIDATION.md).

## README Media and Final Gallery Review

Release 0.2.2 includes a 16-task montage, all 16 complete execution previews, paired future frames at three offsets in two domains, and two paper diagnostic figures. Every task ID 01–16 appears exactly once with its category and original-video link; only task 16 is supplementary. GIFs use 4× playback, width 352, 5 fps and no audio. Static assets remain unmodified.

The review checks the task/source mapping against project-page metadata, frame counts, durations, looping and sampled animation changes on all 16 previews. It also checks all 32 media hashes/source links, README references/captions, desktop/mobile layout and distribution contents. Runtime/test code and training configuration remain unchanged from the audited 0.2.0 implementation. The 39 runtime-test results remain 0.2.0 evidence; this media update does not claim a new training run.

## README Structure Review

Release 0.2.3 places the method figure before all task imagery, follows with attributed paper results and installation/training entry points, and groups optional media into expandable panels. It retains all 16 task previews, original-video links and 32 attributed assets. Default collapsed and fully expanded desktop/mobile layouts are checked separately. The documentation and packaging checks are refreshed; the 39 runtime tests remain evidence from 0.2.0, with runtime/configuration files byte-identical to 0.2.2.
