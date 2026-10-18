# Contributing to RAP-WAM

Thank you for helping make the training code easier to use and inspect. Focused fixes, readable documentation and clearly specified experiments are welcome.

## Start with a Reproducible Case

For a bug, use the issue template with the release, profile, environment, sanitized invocation, exact failure and expected behavior. A small synthetic input is preferable to uploading private demonstrations. For a feature, describe the user workflow and the validation it would need. Do not assign published task performance to an untested configuration.

## Development Setup

Follow [installation](docs/INSTALLATION.md), including the `[data]` extra and FFmpeg/ffprobe. An editable install lets scripts and tests use your current source:

```bash
python -m pip install -e '.[data]'
python scripts/check_docs.py
python -m unittest discover -s tests -v
python -m rap_wam.train --config-name robodojo --cfg job --resolve
python -m rap_wam.train --config-name real_robot --cfg job --resolve
```

The full CPU suite uses synthetic data, reduced models and temporary directories. Record actual failures/skips. Real-model changes additionally need a GPU integration check with auxiliary branches active; the CPU contracts cannot establish 5B convergence or native CUDA behavior.

## Code and Experiment Changes

- Keep package, command, file and configuration names under the public `rap_wam` namespace. Use RAP-WAM for project titles and prose. Use English for repository content and preserve formal scholarly/upstream attribution.
- Preserve explicit tensor/layout contracts. Changes to action ordering, normalization, episode splitting or chunk offsets need corresponding documentation and meaningful contract checks.
- For loss/attention changes, explain the intended mathematics and gradient paths. State changes to compute, sampling, initialization and checkpoint compatibility.
- Record changes to batch, accumulation, warmup and checkpoint selection. New experiments are not automatically matched to the paper protocol.
- Keep external robot integration separate from reusable policy and training interfaces. Do not embed private paths, data, model weights, credentials or machine-specific task launchers.

Prefer small patches with a clear before/after behavior. Follow surrounding code style; avoid unrelated reformatting and dependency changes. Tests should verify consequential behavior rather than duplicate the implementation line for line.

## Documentation and Media

Follow the [Naming and Editorial Guide](docs/STYLE_GUIDE.md): RAP-WAM is the display name; `rap_wam` remains the code namespace.

Use the [documentation index](docs/README.md) to choose the appropriate guide. Keep the README focused on the project, first run and navigation; put detailed field tables and troubleshooting in linked guides. Verify commands against real CLI options and avoid undocumented placeholders for a repository that has not been published.

`python scripts/check_docs.py` checks local Markdown/HTML links, anchors, image references and shell-block syntax. It also checks that README BibTeX matches `citation.bib`. It does not execute every training command or contact external URLs. New media needs an alt description, attribution and a role in explaining the method; do not add model/data archives as documentation assets.

## Release Integrity Checks

`python -m rap_wam.check_release` validates syntax, internal imports, basic publication hygiene and the release manifest. A manifest mismatch after an intentional source edit is expected. After reviewing the changed files, regenerate hashes with `python scripts/update_manifest.py`, then rerun the release checker. This updates `provenance/release_manifest.json` only; it does not certify the change or refresh validation records. Do not suppress checks or present stale validation evidence as current.

The CPU GitHub Actions workflow is included for pushes and pull requests. Its existence is not evidence of a successful hosted run. The local/temporary-environment evidence is described in [VALIDATION.md](docs/VALIDATION.md).

## Pull Request Contents

State the concrete problem, resulting behavior, relevant tests and known limits. Use the provided template. Disclose changes to public defaults and attribution/licensing. Contributions are provided under the applicable project/component license; vendored code retains its existing upstream terms.
