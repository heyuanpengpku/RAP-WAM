# Troubleshooting

[Documentation](README.md) / Troubleshooting

Find the failing stage first: installation → conversion → data/context preflight → model load → training → saving. `--dry-run` checks launch construction, Hydra overrides and supported modes. The standalone preflight checks data; the training entry additionally validates language/context and model configuration, VAE and referenced weight-file presence.

## Installation and Model Assets

| Symptom | Check and action |
|---|---|
| `No module named rap_wam` | Activate the intended environment; run `python -m pip install -e '.[data]'` from the repository root. Check `python -c 'import rap_wam; print(rap_wam.__file__)'`. |
| torch/torchvision import or operator errors | Install the pinned matching pair from the same suitable wheel source. Run `python -m pip check`; avoid mixing unrelated CPU/CUDA builds. |
| CUDA is unavailable | Check the selected Python, GPU allocation and `CUDA_VISIBLE_DEVICES`; compare the installed torch build with the driver. CPU checks do not enable the production training path. |
| `Wan2.2_VAE.pth` / `config.json` missing | Point `WAN_CHECKPOINT_DIR` at the complete original checkpoint directory, preserving its layout. |
| Tensor loading fails after preflight | Basic asset checks do not validate every weight shard. Verify the complete upstream download, file integrity and configuration/weight pairing. |
| UMT5 tokenizer or SentencePiece import fails | Install package dependencies including SentencePiece/protobuf; use the complete tokenizer directory via `--tokenizer` if stored outside the default checkpoint layout. |
| FFmpeg executable/decoder failure | Check both `ffmpeg -version` and `ffprobe -version`; confirm that your build can decode the actual video codec. |

## Data Conversion and Labels

**Existing output directory.** Converters refuse to merge into an existing path. Choose a fresh output directory for a changed conversion. Inspect any previous dataset before deliberately removing it; conversion resume is not implemented.

**Frame-count mismatch.** Inspect all three videos and parquet rows for the episode. The converter requires equality by default. `--truncate-to-shortest` is an explicit truncation option, not a replacement for timestamp synchronization. Report its use as a data-protocol change.

**Unknown or missing task string.** RoboDojo expects exactly one instruction per episode, matched against the ordered vocabulary. Supply a deliberate custom vocabulary to conversion if needed. Regenerate the context table from the emitted vocabulary and set `task.dataset.num_tasks` consistently. Never silently assign a missing label to zero.

**Context table order or dimensions differ.** Use the dataset's exact `task_strings.json`; the saved context `tasks` list must match it in order, and the table must be `[num_tasks,512,4096]` with finite entries. Renaming the file cannot repair a wrong mapping.

**28D data rejected.** Check named arrays, float32/uint8 dtypes, common T, at least 33 frames, image size and episode count. An unspecified 28D concatenated vector is not the raw NPZ schema. The adapter requires the named waist/arm/hand arrays in [DATA_FORMAT.md](DATA_FORMAT.md).

**Unexpected sample count.** An episode with T frames contributes T−32 two-chunk windows before splitting. The loader takes actions at offsets 1–16 and 17–32. A validation split removes whole episodes. Count eligible windows after the split rather than dividing total frames by the action horizon.

## Launch and Training

**Output directory rejected.** `RUN_DIR` must be new on rank zero. Use a fresh run directory even when supplying a resume checkpoint. Check that independent jobs are not sharing one directory.

**Distributed launch waits or fails.** Confirm identical code/dependencies/assets on all nodes, reachable `MASTER_ADDR:MASTER_PORT`, shared paths, one launcher per node, correct `NNODES` and unique `NODE_RANK`. Compare the dry-run summaries. Do not accidentally start multiple independent rank-zero jobs. This release does not provision hosts or test their network fabric.

**CUDA out of memory.** Inspect memory on the intended hardware with the actual resolution and auxiliary branches active. Full 5B DiT optimization and rehearsal can consume substantially more memory than inference or the CPU suite. Batch 1 is an explicit option, but changes local difficulty adaptation. Accumulation changes optimizer timing; it does not lower memory for a single oversized example. There is no validated automatic fallback to LoRA/FSDP or a smaller model.

**No rehearsal/credit at the beginning.** Warmup and random gates can make early participation sparse. Inspect `*_attempted`, `*_succeeded`, `*_applied` and cumulative fields. Branch counters follow local training calls, not exclusively optimizer updates. Use the documented short integration probe to exercise both branches explicitly; retain the production schedule for real training.

**An auxiliary raises an exception.** Read the chained original cause. Strict mode is intentional: the public recipe stops rather than silently disabling a method component. Fix data/model/precision issues before a long run. `policy.strict_auxiliary_errors=false` is an explicit compatibility fallback that warns and logs failures; report the resulting participation and protocol change.

**Loss is finite but success is unknown.** Training loss and synthetic checks are not task-success evaluation. The fixed100 profile has no validation examples; minimum training-loss checkpoints are not selected by robot success.

## Checkpoints and Continuation

A state-dictionary mismatch often indicates a changed architecture/interface or a historical configuration using unavailable module paths. Match the public targets and tensor shapes instead of ignoring missing keys. Keep the RoboDojo language table with the run. Read [CHECKPOINTS.md](CHECKPOINTS.md) before treating a continuation as bit-exact reproduction.

## Reporting an Issue

Include the release, Python/torch/CUDA versions, profile, sanitized command/overrides, dataset dimensions/counts and the original traceback. State whether the failure occurs before model construction, only when an auxiliary runs, or during checkpoint saving. Remove credentials and private infrastructure details. The [bug report template](../.github/ISSUE_TEMPLATE/bug_report.yml) lists the requested fields.
