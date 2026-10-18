# RAP-WAM Documentation

Start with the guide that matches your goal. All shell examples assume the repository root and an activated environment unless a guide says otherwise.

## Train Your First Model

| Step | Guide | You will finish with |
|---|---|---|
| 1 | [Installation](INSTALLATION.md) | Pinned Python dependencies, FFmpeg and a located Wan checkpoint |
| 2a | [RoboDojo Training](ROBODOJO.md) | Converted three-camera demonstrations and an ordered task-context table |
| 2b | [Real-Robot Training](REAL_ROBOT.md) | Validated 28D absolute-joint demonstrations |
| 3 | [Training Recipes](TRAINING_RECIPES.md) | Explicit batch/resources, single-node or multi-node launch |
| 4 | [Configuration Reference](CONFIGURATION.md) | Understood overrides, schedules and loss controls |
| 5 | [Checkpoints and Run Outputs](CHECKPOINTS.md) | Logs, saved states and required companion files |
| When needed | [Troubleshooting](TROUBLESHOOTING.md) | Concrete checks for setup/data/launch failures |

## Understand or Extend the Method

| Guide | Content |
|---|---|
| [Architecture](ARCHITECTURE.md) | Data flow, token groups, policy inheritance and public interfaces |
| [Paper Alignment](PAPER_ALIGNMENT.md) | Equations mapped to implementation, terminology and auxiliary diagnostics |
| [Data Contracts](DATA_FORMAT.md) | Exact LeRobot, NPZ, Zarr and language-table contracts |
| [Reproducibility](REPRODUCIBILITY.md) | Public recipe versus reported experiments, result interpretation and missing components |
| [Naming and Editorial Guide](STYLE_GUIDE.md) | Display name, package identifiers, heading style and evidence wording |
| [Contributing](../CONTRIBUTING.md) | Development checks and expectations for changes |

## Verify a Release

[Validation Scope](VALIDATION.md) describes the CPU suite and its limits. [Release Audit](AUDIT_REPORT.md) records the findings, corrections and evidence; machine-readable evidence and hashes live under [provenance](../provenance/validation.json). [Changelog](../CHANGELOG.md) records user-facing changes.

[Return to the project overview](../README.md).
