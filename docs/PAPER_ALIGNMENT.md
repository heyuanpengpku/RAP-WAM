# Paper and Implementation Guide

Reference: [paper v1, Sections 3 and Appendices A/C/H](https://arxiv.org/html/2610.16443v1). The accompanying [project page](https://rapwam.github.io/) provides the paper and demonstrations. RAP-WAM is the project display name; `rap_wam` is the Python package and source-directory name. Citation metadata preserves the paper's full title.

## Release Scope

The two packaged workflows train the base architecture from target demonstrations. They do not include the agent runner, evidence-guided replay sampler, candidate-policy selection/repertoire, RoboDojo simulator adapter, robot-control stack, evaluation records or pretrained task checkpoints. Inherited model sampling methods are present; an end-to-end deployment/evaluation application is not.

The 28D phone workflow is one physical embodiment/task interface, not the paper's complete 15-task evaluation suite. The paper also describes a 32D interface; it is not supported by this release's strict 28D data adapter. RoboDojo's 35 training instruction labels are distinct from its 42 evaluation task units. Do not describe the vocabulary size as evaluation coverage.

## Paper-to-Code Map

Paths below are relative to the repository root. Internal class/parameter names are kept stable for checkpoints and Hydra configuration.

| Paper concept | Implementation | Configuration / behavior |
|---|---|---|
| Future-supervised queries, Section 3.1 | `rap_wam/policy/imagination_policy.py`: `ImaginationQueryPolicy`, `_build_align_target`, `_compute_align_loss` | 16 queries; detached pooled future targets; 128D projections; cosine alignment weight 0.1 |
| Directional attention, Section 3.1 | `rap_wam/policy/decoupled_policy.py`: `_build_decouple_attn_mask`, `_joint_self_attention` | Action rows cannot read video keys; state and query paths remain. Video rows can read action keys. Same mask during sampling. |
| Coupled Action Projection (CAP) | `rap_wam/policy/residual_policy.py`: `ResidualActionHead` | Sum of compact MLP and time-modulated sequence head; MLP hidden fraction 0.25 and zero output initialization. `coarse_*`/`residual_*` are parameter names, not separate prediction targets. |
| Predictive rehearsal, Section 3.2 | `rap_wam/policy/cross_chunk_policy.py`: `compute_loss`; `difficulty_adaptive_policy.py`: `_ccrt_chunk1_loss` | `ccrt_*`: probability max 0.25, four gradient-free sampling steps, supervised recorded second chunk. No extra query-alignment loss in this auxiliary. |
| Scheduled state substitution | `rap_wam/policy/on_policy_state_policy.py`: `_opsr_p`; `difficulty_adaptive_policy.py`: `_ccrt_chunk1_loss` | `opsr_p_max=0.5`; convert last predicted raw absolute-joint action into a normalized state. Optional local-batch difficulty correction vanishes at batch 1. |
| Spatial rehearsal-error prior P | `rap_wam/policy/rap_policy.py`: `_ccrt_chunk1_loss`, `_update_uprime` | `_uprime_ema`: channel/batch/time-averaged error, adaptive pooling to 14x14, first valid map copied directly, then EMA decay 0.9. `uprime_use_gap=false`. |
| Matched perturbation response S and current error E, Section 3.3 | `rap_wam/policy/rap_policy.py`: `_puga_noise`, `_puga_forward_video`, `_puga_loss` | Fresh auxiliary noising, shared diffusion level; same video noise/conditioning between the gradient reference and no-grad action-perturbed forward. Normalized action perturbation std 0.1. |
| Visual/temporal credit | `rap_wam/policy/rap_policy.py`: `_puga_norm01`, `_puga_loss` | Detached source maps, per-sample normalization with floor 1e-6. Visual coefficients use P/S/E; temporal coefficients pool raw S before time normalization. |
| Language-conditioned RoboDojo | `rap_wam/policy/language_policy.py`; `rap_wam/task_context.py` | Precomputed UMT5 embeddings selected by episode task label; keep ordered table with the checkpoint. |
| Joint flow matching and sampling, Appendix C.1 | `rap_wam/core/policy/wan_joint_action_policy.py`; `imagination_policy.py`; `decoupled_policy.py` | Independent video/action noise at a shared level; velocity targets noise minus clean input. Observed visual prefix is clamped. |

The inherited chain is an implementation decomposition of one policy, not a multi-policy ensemble. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Credit Coefficients and Loss Reduction

With `uprime_use_gap=false`, internal U/L/D denote the paper's P/S/E. The video weight is `1 + N(P) + N(S) + 0.5*N(E)`, between 1 and 3.5. N separately min-max normalizes each sample over future time and space, with a denominator floor of 1e-6.

The temporal weight is `1 + N_time(mean_hw(S))`, between 1 and 2. The spatial prior broadcasts identically across time, so its temporal normalization is zero; prediction error does not enter the action branch. Repeat each of the four future-latent time weights four times for 16 action steps. The code uses ceiling-repeat/truncation for other compatible horizons. Do not average spatially normalized S to obtain these weights.

The extra video and action losses divide weighted sums by their expanded weight sums, with coefficients `puga_weight=0.5` and `puga_act_weight=0.3`. Their warmup multiplier and gate probability (`puga_p=0.5`) apply in addition to the ordinary base loss. The same reference prediction carries gradients through the squared errors; coefficient construction and the perturbed forward do not. This response is sensitivity of model outputs to action inputs, not measured physical causal influence.

`puga_w_unc`, `puga_w_lev`, `puga_w_diff` map to the P/S/E coefficients (1/1/0.5). `puga_n_samples` must be 1; the training entry rejects other values rather than silently ignoring them. No additional scoring network is constructed. Rehearsal, alignment targets and response probing are training operations and do not run during ordinary action sampling.

## Recipe Differences from the Reported Base

| Setting | Paper reference | Packaged public default |
|---|---|---|
| RoboDojo per-device / global batch | 1 / 96 | 2 / resource-dependent |
| Physical per-device / global batch | 1 / 120 | 2 / resource-dependent |
| Gradient accumulation | 1 in archived reference runs | 1 |
| RoboDojo split | 3,499 train / 1 validation episode | 5% episode holdout of supplied data |
| RoboDojo VAE work | Recorded futures cached, predicted inputs encoded live | Online encoding; latent cache disabled |
| Physical input | Task-dependent 28D or 32D | Only 28D head-camera fixed100 |
| Checkpoint selection | Archived experiment protocol | Synchronous epoch checkpoints, latest + top 3 by training loss |
| Auxiliary exceptions | Only successfully computed auxiliaries participate | Exceptions raise by default; sampled/success/applied/failure counters logged |

Core defaults retain BF16, full-parameter DiT training, frozen VAE, no LoRA, no parameter EMA, AdamW at 2e-4, observation/future/action lengths 1/16/16, and auxiliary warmups 18,000 (RoboDojo) / 6,000 (physical). Recipe epoch budgets and optimizer warmup follow the archived source configs; the paper's Table 2 does not specify every such value. Module counters count local training calls, including accumulated minibatches; they are not a global sample budget or a strict optimizer-update counter when accumulation changes.

For an explicit archived-batch comparison, set `BATCH_SIZE=1`. Reproducing global batch 96 or 120 additionally requires the corresponding resource product, or an explicitly documented accumulation choice. Accumulation changes update timing and does not recreate across-sample local difficulty variation. The public 5% split/cache/checkpoint differences still apply. This release does not provide an exact-paper reproduction preset or transfer published scores to new data/training settings.

## Inference Settings Are a Separate Protocol

Appendix A specifies RoboDojo evaluation with 30 denoising steps, eight executed actions per replan, Savitzky-Golay smoothing (window 9, polynomial order 2) and four-action boundary blending. The packaged policy's `num_inference_steps=16` is a training/configuration default; rehearsal explicitly overrides it with four steps. The evaluation wrapper implementing smoothing, action execution and replanning is not bundled. A deployer must supply that integration and the correct joint mapping, units and control rates. Changing the sampling-step count alone does not reproduce the evaluation protocol.

## Auxiliary Participation and Failures

Public recipes explicitly enable `policy.strict_auxiliary_errors=true`. A raised rehearsal/credit exception stops training, preserving the original exception as its cause. An explicit `policy.strict_auxiliary_errors=false` restores the inherited exception fallback, now with a warning and failure counter. It may produce fewer effective auxiliary updates and should be reported as a protocol difference.

For each update, `rehearsal_*` and `credit_*` expose `attempted`, `succeeded`, `applied` and `failed`. Attempted means the sampling gate opened; succeeded means a loss tensor was returned; applied additionally requires a positive warmup multiplier. Failed counts caught exceptions, not probability skips. A helper returning no loss gives attempted=1, succeeded=0 without an exception count. The corresponding `*_total` fields accumulate within each process and are checkpointed with the rank-zero policy; they are not synchronized across DDP ranks. Epoch summaries retain the final cumulative value rather than its average. Compute participation as applied updates divided by observed training updates; attempts alone do not establish participation. `total_loss` includes the applied credit term.

Missing/invalid data should first fail the strict preflight. The diagnostics complement it; they do not validate convergence or prove that a public run matches historical branch frequencies. See [VALIDATION.md](VALIDATION.md) for actual checks and remaining external prerequisites.
