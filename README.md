# RAP-WAM

**Learning World-Action Models through Predictive Rehearsal and Action Perturbations**

[Project page](https://heyuanpengpku.github.io/RAP-WAM/) · [Manuscript and appendix](https://heyuanpengpku.github.io/RAP-WAM/assets/paper/rap-wam-public.pdf)

Yuanpeng He, Ceyao Zhang, Fangjing Li, Kefei Zhu, Lijian Li, Yueheng Li, Tianjia He, Wenpin Jiao, Zhi Jin, Yaodong Yang, and Yuanpei Chen.

Peking University · Beijing Jiaotong University · University of Macau · Psibot

This repository contains the research project website. It is not a release of the training implementation.

## Website

A responsive English/Chinese project page with:

- An overview reel covering all 16 real-robot task executions, mostly at 4× speed; the final phone-box sequence slows to 2× and then 1× so the finishing lid touch remains visible. Individual recordings retain original-speed playback.
- Six-method real-robot comparisons, task selection, and downloadable CSV data.
- Base/system comparisons across five RoboDojo capabilities and 42 searchable task results.
- Interactive paired recorded/predicted training-frame comparisons for all 51 available examples (16 real-robot and 35 simulation), with domain filters, task search, thumbnails, previous/next controls, and five future offsets.
- The architecture overview, manuscript, authors, and BibTeX citation.

The shared real-robot comparison covers tasks 01–15 (89.00% task-macro SR). Phone-box assembly is supplementary and does not enter that average. Simulation results distinguish the fixed base from the task-conditioned post-trained system. Prediction metrics describe training samples, separate from execution success.

## Run locally

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000`. No build step or JavaScript dependency installation is needed.

GitHub Pages serves the repository root on the `main` branch. `.nojekyll` enables direct static publishing. All site assets use relative URLs and support the `/RAP-WAM/` project path.

## Content maintenance

- `index.html`: bilingual page text, author links, and citation.
- `app.js`: interactive views and language switching.
- `styles.css`: responsive visual design.
- `data/results.json`: manuscript-aligned result values; absent supplementary baseline values remain `null`.
- `assets/videos/`: H.264 MP4, original-speed full recordings with audio removed. `all-tasks-camera-clean-scored-1080p60.mp4` combines all 16 task executions with opening/closing graphics and a slower phone-box finale. Videos use progressive-download fast-start metadata.
- `assets/images/`: original video posters and exported manuscript figures/paired prediction frames. No generative image editing is used.
- `assets/paper/rap-wam-public.pdf`: public English manuscript with author attribution and clearly marked appendix.
- `assets/fonts/`: self-hosted Inter and Space Grotesk, with SIL Open Font Licenses.

An arXiv link can be added when available.

The homepage identifies Peking University as the leading institution. Its unaltered red emblem is from the [official PKU visual identity download center](https://vim.pku.edu.cn/xzzq/) (circular emblem archive). Future latents are shown as a schematic token grid, and the supplementary phone-box recording is centered. Individual task recordings have zero audio streams; the homepage overview has an optional instrumental soundtrack.

Author affiliations: Yuanpeng He, Wenpin Jiao, Zhi Jin and Yaodong Yang — Peking University; Fangjing Li — Beijing Jiaotong University; Lijian Li — University of Macau; Ceyao Zhang, Kefei Zhu, Yueheng Li, Tianjia He and Yuanpei Chen — Psibot. The public manuscript uses superscript institution indices.

Fruit placement has its 7.1-second stationary introduction removed in both the individual video and the overview reel. Burger-box assembly now includes a verified final-step training GT/prediction visualization; its SSIM and LPIPS are computed from the five saved RGB pairs, and this basis is disclosed in the viewer.

The homepage plays one continuous overview video containing all 16 tasks. Individual-task controls appear only in the robot demo gallery.

The homepage overview cuts each scene after object placement/release, omits subsequent return-to-home tails, and joins scenes with 0.25-second cross-dissolves. The gallery recordings remain separate.

The method explorer defines Coupled Action Projection (CAP), illustrates its additive compact-MLP/noise-conditioned readout, and presents predictive rehearsal as generated-condition training plus a persistent-error prior.

Towel organization uses the newly supplied 25-second recording in the gallery and homepage overview. Its poster is refreshed from the same recording; the overview ends this task at object placement/release and retains the short cross-dissolve edit. Individual task recordings remain silent.

The overview preserves the phone-box assembly through the finishing touch (source 58.4 seconds), then adds a blue/red ripple and closing title. A matching opening title and a short dissolve form the loop. These are presentation overlays; individual execution recordings remain unchanged.

The overview is encoded at 1920×1080 and 60 fps, rebuilt from source recordings. Opening/closing typography uses the same self-hosted project fonts, rendered at 3840×2160 before antialiased downsampling; the touch ripple is also supersampled. The web encoding uses H.264 CRF 20 with a 5 Mbps bitrate cap and fast-start metadata.

The 60 fps overview samples 15 original frames per source second in its 4× portions, versus six in the earlier 24 fps edit. It uses recorded frames rather than motion-generated intermediate views. The phone-box finale retains its 2× and 1× sections, and scene/loop transitions are resampled at the new output frame rate.

Each scene begins after the first 0.6 seconds of its recording, removing the recording-button camera disturbance. Fruit placement retains its previously trimmed 7.1-second start. Scene cuts use continuous footage with a cosine dissolve; the overview runs at 4× with the existing 2×/1× phone-box finale.

Homepage music: [Electric Dreams](https://www.scottbuckley.com.au/library/electric-dreams/) by [Scott Buckley](https://www.scottbuckley.com.au/), used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The excerpt is trimmed, normalized and faded, and is included only in the overview video. The page provides a Music on/off control; autoplay starts muted. Audio follows video pause, seeking, fullscreen and the loop. License/credit details are in `assets/videos/music-attribution.txt`.

The same recording-start trim is applied to the individual silent recordings (except the already-trimmed fruit clip). Source recordings are preserved. The earlier entry speed ramps are removed, and soundtrack fades and the finishing-touch effect are aligned to the shorter overview.
