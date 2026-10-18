# Reproducibility and Release Scope

[Documentation](README.md) / Reproducibility

The repository makes the base training implementation usable with public module paths and strict data contracts. It does not independently reproduce the full experimental system in [paper v1](https://arxiv.org/html/2610.16443v1).

## Three Distinct Objects

| Object | Meaning | Provided here |
|---|---|---|
| Base learner | Future-supervised queries, CAP, predictive rehearsal and response-credit learning | Model/loss implementations and two data/training interfaces |
| Public training recipe | A portable configuration over data and assets supplied by the user | Hydra profiles, converters, checks and DDP launchers |
| Reported research system | Trained policies plus agent-guided post-training, task-conditioned retention and evaluation | Paper/project-page description; the complete system is not bundled |

A successful software test checks a contract or an integration path. It does not establish matching convergence, branch participation frequencies, selected checkpoints or task performance.

## Differences That Change a Reproduction Attempt

The [paper-to-code comparison](PAPER_ALIGNMENT.md#recipe-differences-from-the-reported-base) is the detailed reference. In particular:

- Reported local/global batch is 1/96 for RoboDojo and 1/120 for physical training. Public defaults use local batch 2 with resource-dependent global batch and accumulation 1.
- Reported RoboDojo training uses 3,499 train episodes and one validation episode. Public RoboDojo defaults hold out 5% of supplied episodes, with deterministic episode isolation.
- Reported RoboDojo recorded-future VAE latents were cached. Public profiles encode online and disable the optional cache path; no offline cache builder is bundled.
- Checkpoint writing/selection follows the public synchronous workspace and training-loss top-k rule rather than the archived asynchronous execution schedule.
- Auxiliary exceptions raise by default; the explicit tolerant mode logs a fallback. Effective participation depends on schedules and successful calculations, not only configured probabilities.
- The physical adapter is strictly the 28D head-camera fixed100 workflow. The paper also contains other tasks and a 32D interface.

Setting local batch to 1 alone does not resolve all differences. Accumulating batch-1 gradients does not create local-batch difficulty variation. Full DiT training is enabled; LoRA is not part of the validated public recipe.

## Interpreting Paper Results

The paper evaluates a fixed base and a task-conditioned system on 42 RoboDojo task units. Generalization variants are combined within their parent task. Task metrics are averaged within each of five capabilities; those capability means receive equal weight. An unweighted mean across all episodes is a different metric.

The complete system retains one policy per task using evaluation evidence; it is not the output of a single universal improved checkpoint or parameter averaging. The physical 89.00% figure is a macro-average over 15 scored tasks, not a performance claim for the one training workflow packaged here. Training-window image diagnostics on the project page are also separate from task-success evaluation.

## What Is Needed for a Full Independent Study

Use the corresponding data distribution, trained checkpoints or a fully specified training rerun, verified preprocessing/normalization, simulator/robot integration and evaluation metadata. A complete post-training study additionally needs execution evidence, replay-pool construction, candidate generation, task-policy retention rules and their evaluation records. These are not recreated by enabling an extra config flag in this repository.

For a new experiment, retain at least:

| Record | Include |
|---|---|
| Code and environment | Release/version, local modifications, installed versions, driver/CUDA details |
| Assets | Original model identity and exact files; tokenizer/context vocabulary order |
| Data | Episode identities/counts, frame counts, conversion options, coordinate conventions |
| Split | Seed, ordered source episodes, held-out episode identities |
| Optimization | Local/global batch, accumulation, node/GPU layout, LR and schedules |
| Auxiliary behavior | Actual participation/failure counts and any tolerant fallback |
| Selection | Candidate checkpoints and the criterion used to retain one |
| Evaluation | Tasks, seeds/layouts, valid episode counts, sampling and action-processing protocol |

[Validation Scope](VALIDATION.md) records the checks performed on this release. [Checkpoints](CHECKPOINTS.md) explains saved state and continuation limits.

## Corrections in Release 0.2.0

Training-only normalization, dark-uint8 image scaling, explicit language labels, bounded-loader accumulation, scheduler budgeting and checkpoint continuation were corrected after the repository audit. These changes can affect newly trained policies. They do not establish reproduction of previously reported results. Existing checkpoints retain saved normalization; legacy counters/epoch state have only the compatibility behavior described in [CHECKPOINTS.md](CHECKPOINTS.md).
