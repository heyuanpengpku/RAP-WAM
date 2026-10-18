# RAP-WAM Changelog

## 0.2.3 — 2026-10-18

### README Structure and Navigation

- Lead with the method figure, research contribution and reported results.
- Put installation and both training workflows ahead of optional visual galleries; link complete procedures through the documentation index.
- Preserve all 16 task previews and original videos in four expandable groups, and fold prediction/diagnostic figures into labeled panels.
- Verify collapsed and expanded desktop/mobile layouts, media coverage, references and source artifacts. Runtime code and training configuration are unchanged.

## 0.2.2 — 2026-10-16

### Complete Robot Demonstration Gallery

- Include all 16 project-page robot tasks in task order, each with a complete 4× GIF preview, category and original-speed video link.
- Identify phone-box assembly as supplementary and preserve the 15-task metric boundary.
- Use consistent numbered media filenames and compact encoding; verify the complete 01–16 task set and source mapping.
- Recheck the remaining README, documentation references, attribution, responsive layout and release contents. Runtime code/configuration remains unchanged.

## 0.2.1 — 2026-10-16

### README Research Showcase

- Reorganize the README around robot demonstrations, the method, predicted futures and supervision diagnostics before the training guides.
- Add the project-page 16-task montage and four complete execution clips converted to compact, looping 4× GIF previews.
- Expand recorded/predicted comparisons to three future offsets in both the physical and RoboDojo domains.
- Include response-allocation and credit-dynamics figures with source attribution and clear experimental scope.
- Preserve runtime code/configuration and the 0.2.0 training audit; verify media integrity, animation, documentation links and source distribution contents.

## 0.2.0 — 2026-10-16

### Training and Workflow Corrections

- Fit normalization on training episodes only; preserve low uint8 RGB intensities.
- Require explicit language labels in training and sampling, including workspace diagnostics.
- Validate vocabularies, per-episode labels, cross-shard layouts, model assets and supported launch options.
- Compose Hydra overrides during dry-run, keep output paths consistent and reserve new run directories atomically.
- Bound the actual loader so shortened runs flush accumulated gradients; budget scheduler steps from the resulting optimizer updates.
- Reject missing resume paths and nonfinite losses. Save checkpoints atomically, preserve the chosen output directory and continue from the next completed epoch.
- Persist auxiliary schedules/prior readiness/counts, retain legacy model-state loading, and preserve sampled metrics for checkpoint selection.
- Validate a fresh Python 3.10 CPU installation, 39 runtime tests, two-process CPU DDP and native CUDA projection heads; full 5B and robot evaluation remain external.

## 0.1.5 — 2026-10-16

### Project Identity and Editorial Consistency

- Use RAP-WAM for the README, documentation landing pages, citation title and package description; retain `rap_wam` for importable and machine-readable identifiers.
- Standardize heading capitalization and repair navigation labels/anchors, image captions and workflow descriptions.
- Add a concise naming/editorial guide and automatic display-name checks for future changes.
- Refresh documentation previews and distribution metadata without changing the training algorithms.

## 0.1.4 — 2026-10-16

### Documentation and Repository Usability

- Rebuilt the project README with method overview, attributed figures, paper-result context, workflow comparison, first-run commands, repository navigation and full citation.
- Added installation, RoboDojo, real-robot, configuration, checkpoint, reproducibility and troubleshooting guides, linked through a documentation index.
- Added a named-array export example, environment templates, contribution guidance and issue/pull-request templates.
- Added a documentation checker and CPU GitHub Actions workflow. Hosted execution requires publishing the repository and running Actions; it is not claimed by this local release.
- Corrected release checking to exclude local virtual environments, model assets, data and outputs; the runtime suite now has 20 tests.
- Preserved the two training profiles and model/loss behavior from 0.1.3. Updated source/package publication artifacts and validation records.

## 0.1.3 — 2026-10-16

- Added paper/project metadata, BibTeX and CFF citations, and an explicit paper-to-code mapping.
- Made auxiliary branch exceptions fatal by default with an explicit tolerant override; added participation/failure counters and corrected total-loss diagnostics after credit.
- Added numerical and failure-handling contracts, bringing the runtime suite to 19 tests.
- Documented differences between public defaults and paper runs, and excluded unbundled post-training/evaluation systems from reproduction claims.

## 0.1.2 — 2026-10-16

- Consolidated the public namespace and source layout as `rap_wam`.
- Completed dependency/tokenizer checks, strict data preparation and portable training entrypoints for RoboDojo and the 28D fixed100 workflow.
- Validated reduced-model policy and CPU training/checkpoint contracts; preserved upstream licensing and release provenance.
