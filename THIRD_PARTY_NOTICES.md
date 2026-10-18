# Third-Party Notices

RAP-WAM project code and release tooling are offered under the MIT license in `LICENSE`. This does not replace the licenses of the following components or grant redistribution rights to datasets or model weights.

- **Wan2.2**: `third_party/wan_official/wan/` is a minimal vendored runtime from [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2). Copyright notices are preserved. Its Apache-2.0 license is included at `third_party/wan_official/LICENSE.txt`. The package initializer was reduced to the required modules in the original training snapshot; this release additionally defers T5's default CUDA device query until object construction. Model weights must be obtained separately under their applicable terms.
- **Diffusion Policy**: the normalization/module/checkpoint conventions and portions of the policy utilities derive from [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy). The upstream MIT license is retained at `third_party/licenses/diffusion_policy.txt`.
- **T5**: the vendored Wan T5 implementation identifies its origin in Hugging Face Transformers; upstream author notices in the source are retained.
- PyTorch, Accelerate, Diffusers, Transformers, Hydra, Zarr, and other dependencies are installed separately and retain their own licenses.

The RoboDojo dataset, private real-robot demonstrations, pretrained model weights, and benchmark simulator are not bundled in this code release. Their names identify interfaces and data formats; no benchmark assets are relicensed by this MIT license.
