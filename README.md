# RAP-WAM

**Learning World-Action Models through Predictive Rehearsal and Action Perturbations**

[Project page](https://heyuanpengpku.github.io/RAP-WAM/) · [Manuscript and appendix](https://heyuanpengpku.github.io/RAP-WAM/assets/paper/rap-wam-public.pdf)

Yuanpeng He, Ceyao Zhang, Fangjing Li, Kefei Zhu, Lijian Li, Yueheng Li, Tianjia He, Wenpin Jiao, Zhi Jin, Yaodong Yang, and Yuanpei Chen.

Peking University · Beijing Jiaotong University · University of Macau · Psibot

This repository contains the research project website. It is not a release of the training implementation.

## Website

A responsive English/Chinese project page with:

- An overview reel covering all 16 real-robot task executions at 4× speed, with continuous playback; individual recordings retain original-speed playback.
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
- `assets/videos/`: H.264 MP4, original-speed full recordings with audio removed. `all-tasks-polished.mp4` combines all 16 task executions at 4× speed. Videos use progressive-download fast-start metadata.
- `assets/images/`: original video posters and exported manuscript figures/paired prediction frames. No generative image editing is used.
- `assets/paper/rap-wam-public.pdf`: public English manuscript with author attribution and clearly marked appendix.
- `assets/fonts/`: self-hosted Inter and Space Grotesk, with SIL Open Font Licenses.

An arXiv link can be added when available.

The homepage identifies Peking University as the leading institution. Its unaltered red emblem is from the [official PKU visual identity download center](https://vim.pku.edu.cn/xzzq/) (circular emblem archive). Future latents are shown as a schematic token grid, and the supplementary phone-box recording is centered. All 17 MP4 files have zero audio streams.

Author affiliations: Yuanpeng He, Wenpin Jiao, Zhi Jin and Yaodong Yang — Peking University; Fangjing Li — Beijing Jiaotong University; Lijian Li — University of Macau; Ceyao Zhang, Kefei Zhu, Yueheng Li, Tianjia He and Yuanpei Chen — Psibot. The public manuscript uses superscript institution indices.

Fruit placement has its 7.1-second stationary introduction removed in both the individual video and the overview reel. Burger-box assembly now includes a verified final-step training GT/prediction visualization; its SSIM and LPIPS are computed from the five saved RGB pairs, and this basis is disclosed in the viewer.

The homepage plays one continuous overview video containing all 16 tasks. Individual-task controls appear only in the robot demo gallery.

The homepage overview cuts each scene after object placement/release, omits subsequent return-to-home tails, and joins scenes with 0.25-second cross-dissolves. The gallery recordings remain separate.
