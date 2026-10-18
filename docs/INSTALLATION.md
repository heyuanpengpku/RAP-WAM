# Installation

[Documentation](README.md) / Installation

## Environment and Assets

| Component | Requirement / role |
|---|---|
| Training OS / device | Linux with an NVIDIA CUDA GPU; multi-process training uses ordinary DDP |
| Python | 3.9–3.12 in package metadata; Python 3.10 recommended for the provided Linux lock |
| PyTorch / torchvision | 2.8.0 / 0.23.0, built for a compatible CUDA driver |
| Precision | BF16 in both public training profiles |
| Video tools | FFmpeg and ffprobe on `PATH`; decoder support for the source videos, including AV1 when used |
| Backbone | Complete Wan2.2-TI2V-5B transformer and VAE files |
| RoboDojo text preparation | UMT5 encoder checkpoint and tokenizer, used before training |
| Data | Compatible demonstrations, converted or already in the documented Zarr format |

Full 5B DiT training includes optimizer state, attention activations, video latents and auxiliary forwards. No minimum-memory or one-GPU fit claim has been validated for these defaults. A launch example describes a command, not a measured resource requirement. Do not treat the reduced CPU tests as a hardware sizing benchmark.

## Install the Training Environment

Extract/check out the repository and open its root directory. No public source-repository URL is assumed by these instructions.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Example CUDA 12.8 build; choose a build compatible with your driver.
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[data]'
```

Official PyTorch 2.8.0 wheels also provide CUDA 12.6 and 12.9 builds; consult [PyTorch's version-specific installation instructions](https://pytorch.org/get-started/previous-versions/#v280). Keep the paired torch/torchvision versions unchanged unless you validate an intentional dependency update.

On an Ubuntu/Debian host, video tools can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ffmpeg -version
ffprobe -version
```

The `[data]` extra installs pandas and pyarrow for LeRobot conversion. Optional Weights & Biases support uses `python -m pip install -e '.[data,tracking]'`; local logging is available without it and tracking is disabled by default. FlashAttention is not a mandatory dependency: the vendored attention layer has a PyTorch fallback. Native GPU kernel behavior still needs verification on the target machine.

## Resolved Linux Dependencies

`requirements-linux-py310.lock` records the resolved training-plus-data Python graph for Linux/Python 3.10. It is a version lock, not a binary-wheel hash lock, CUDA driver specification or system-package lock. The torch/torchvision build must still match your device environment.

```bash
# After installing the appropriate pinned torch / torchvision build:
python -m pip install -r requirements-linux-py310.lock
python -m pip install -e . --no-deps
python -m pip check
```

Use `pyproject.toml` on other supported Python versions. Do not mix an unrelated environment's transitive pins into a release report without recording the differences.

## Prepare Model Files

Obtain the complete [Wan2.2-TI2V-5B checkpoint](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) using your preferred Hugging Face download mechanism. Keep its directory structure intact. A minimal training layout contains:

```text
assets/Wan2.2-TI2V-5B/
├── config.json                        # or config under transformer/ or model/
├── ... transformer weight files ...
├── Wan2.2_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth     # RoboDojo context preparation
└── google/umt5-xxl/                   # tokenizer files for context preparation
```

After installing this package, its Hub dependency can download a complete snapshot:

```bash
python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Wan-AI/Wan2.2-TI2V-5B",
                  local_dir="assets/Wan2.2-TI2V-5B")
PYTHON
```

This downloads model assets separately from the code and can require substantial disk/network capacity. Retain the model revision/file identity for an experiment. See the [Hub download documentation](https://huggingface.co/docs/huggingface_hub/guides/download).

The exact transformer weight filenames/layout follow the upstream checkpoint. Preflight checks nonempty VAE/weight files, the shared transformer configuration directory and all referenced weight shards. It does not deserialize the tensors; model construction still validates their content.

```bash
export WAN_CHECKPOINT_DIR="$PWD/assets/Wan2.2-TI2V-5B"
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
python -m rap_wam.train --config-name robodojo --cfg job --resolve
```

For a tokenizer stored elsewhere, pass `--tokenizer /path/to/umt5_tokenizer` to `rap_wam.task_context`. Context preparation loads UMT5 once; the RoboDojo training loop reads the resulting context table without loading the text encoder. The 28D real-robot recipe uses no task-language table.

## CPU-Only Validation Environment

CPU verification can inspect configuration, convert synthetic data and run the reduced-model test suite. It cannot train the production CUDA/BF16 path as a substitute for GPU validation.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[data]'
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
python scripts/check_docs.py
python -m rap_wam.check_release
python -m unittest discover -s tests -v
```

Install FFmpeg/ffprobe first to exercise the video conversion test. Without them, that test reports a skip; a partial suite should not be reported as complete conversion validation.

Next: [RoboDojo](ROBODOJO.md) or [real robot](REAL_ROBOT.md). Setup errors: [troubleshooting](TROUBLESHOOTING.md).
