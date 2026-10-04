# WorldModelScope

A fair, scale-free benchmark for comparing visual world models on robot manipulation video.

## Motivation

A **world model** is a neural network that learns to predict how a scene will change: given a
short history of camera frames and the action a robot is about to take, it predicts the latent
representation of the next frame. Several such models now exist — V-JEPA 2, V-JEPA 2.1,
DINO-WM, LeWorldModel, among others — each built on a different encoder backbone and,
critically, each predicting in its own private latent space.

This is the core problem WorldModelScope addresses: **the training loss reported by each model
is measured on a different, incomparable scale**, so raw numbers cannot be placed side by side.
Worse, a model can post a deceptively good score simply by learning a latent space that barely
changes from one frame to the next, collapsing the prediction task rather than solving it.
WorldModelScope evaluates all models on the same video clips, using only metrics that do not
depend on any model's internal scale.

## Models Compared

| Model | Encoder | Latent per frame | Action-conditioned |
|---|---|---|---|
| **V-JEPA 2.1** | Frozen V-JEPA 2.1 video ViT-B | 8 × 196 × 768 | Yes *(action-conditioned predictor implemented in this project — not part of the public release)* |
| **DINO-WM · DINOv2** | Frozen DINOv2 image ViT | ~64 × 384 | Yes |
| **DINO-WM · EUPE** | Frozen EUPE image ViT | ~196 × 384 | Yes *(EUPE variant implemented in this project — original release uses DINOv2 only)* |
| **LeWorldModel** | ViT-S/14, trained from scratch | 1 × 384 | Yes |

Three things were built specifically for this benchmark rather than reused as-is:
- **V-JEPA 2.1's action-conditioned predictor** — the public checkpoint ships only a dense
  visual encoder with no predictor able to consume the robot's action. A frame-causal,
  action-conditioned predictor was implemented and trained on top of it, following the recipe
  originally introduced for V-JEPA 2-AC.
- **A second DINO-WM variant using EUPE** (Efficient Universal Perception Encoder) in place of
  the original frozen DINOv2 backbone, to test whether the choice of frozen visual backbone
  changes the model's behaviour. Predictor, training procedure, and planning objective are kept
  identical between the two variants.
- **LeWorldModel retrained from scratch on larger-scale, harder data.** Unlike the other three
  models, LeWorldModel has no frozen or pretrained encoder — its encoder and predictor are
  trained jointly, end-to-end, from raw pixels. Because of this, the data it is trained on
  matters more directly to what it actually learns than it does for the frozen-encoder models.
  It was retrained on the same larger-scale, more dynamically demanding data used throughout
  this benchmark (BridgeData V2 and NVIDIA's Cosmos synthetic robot video) rather than
  whatever smaller-scale data the original release used, specifically so the model would be
  pushed to learn more of the scene's actual dynamics rather than settling for a
  near-static latent shortcut.

## Datasets

- **[BridgeData V2](https://rail-berkeley.github.io/bridgedata/)** — real robot manipulation
  trajectories on a WidowX arm: 60,000+ trajectories across 13 skills and 24 environments.
  Primary dataset for all quantitative results below.
- **[NVIDIA Cosmos](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes)** —
  large-scale synthetic robot video covering contact-rich manipulation, collisions, and
  humanoid motion, used to stress-test rollout stability beyond BridgeData V2's tabletop
  pick-and-place setting.

All four models are evaluated at training step 16,000, on the held-out validation slice of
BridgeData V2 (80 shards, one 16-frame clip per episode, 4 fps, 224 px, fixed seed 0).

## Methodology

### Latent-space metrics

| Metric | What it measures |
|---|---|
| **NMSE** | Teacher-forced one-step prediction error, normalized by target variance. Not comparable across models on its own. |
| **Relative L2** | Robustness cross-check on NMSE, per-sample normalized. |
| **Rollout NMSE(h)** | One-step NMSE under open-loop (autoregressive) rollout at horizon *h*. |
| **Drift ratio(h)** | `NMSE(h) / NMSE(1)` — isolates how fast error *compounds*, independent of one-step accuracy. Directly comparable across models. |
| **Path straightness** | Ratio of rollout trajectory curvature to real trajectory curvature — detects collapse toward an averaged, near-linear rollout. |
| **Action reliance** | `(NMSE with shuffled actions − NMSE with real actions) / NMSE with shuffled actions` — whether the prediction actually depends on the commanded action. |
| **Effect vs. zero** | Same test against zeroed rather than shuffled actions — distinguishes "reacts to *an* action" from "reacts to *which* action." |

### Pixel-space evaluation (this project's core contribution)

A lightweight decoder $D_i$ is trained per model, mapping its latent space back to RGB pixels.
Each decoder is trained for 15,000 steps on BridgeData V2, reconstructing the ground-truth
frame from the **encoder's own latent** $z_t$ — never from the predictor's latent. Once
trained, the decoder is frozen and applied, unmodified, to the predictor's latent $\hat z_t$ at
evaluation time.

This produces two reconstructions per frame:

```
ceiling  = m(D_i(z_t),     x_t)        # decoder's own error alone
penalty  = m(D_i(ẑ_t),    x_t) - ceiling   # predictor's error, decoder blur cancelled out
```

for pixel metrics `m` ∈ {LPIPS, PSNR, SSIM}. Because both reconstructions pass through the
identical, frozen decoder, any additional error in the penalty is attributable to the
predictor alone — not to the decoder having been trained differently in the two cases.

## Results

### One-step prediction accuracy (teacher-forced)

| Model | NMSE ↓ | Relative L2 ↓ |
|---|---|---|
| V-JEPA 2.1 | 0.233 | 0.268 |
| DINO-WM · DINOv2 | 0.113 | 0.255 |
| DINO-WM · EUPE | 0.089 | 0.201 |
| **LeWorldModel** | **0.005** | **0.056** |

> Not directly comparable across models — a latent that naturally moves less between frames
> yields lower NMSE without the predictor being any better. See rollout/action metrics below.

### Open-loop rollout stability (horizon h = 4)

| Model | NMSE(h=4) | Drift ratio | Path straightness |
|---|---|---|---|
| V-JEPA 2.1 | 0.508 | 2.18 | **0.939** |
| DINO-WM · DINOv2 | 0.241 | **2.13** | 0.646 |
| DINO-WM · EUPE | 0.198 | 2.22 | 0.562 |
| LeWorldModel | 0.035 | 6.95 | 0.240 |

LeWorldModel's drift ratio is 3× higher than the others; its rolled-out trajectory is ~4×
straighter than the real one, indicating collapse toward a near-linear average path rather than
genuine tracking.

### Action grounding

| Model | Action reliance ↑ | Effect vs. zero ↑ |
|---|---|---|
| V-JEPA 2.1 | 0.015 | 0.013 |
| **DINO-WM · DINOv2** | **0.219** | **0.127** |
| DINO-WM · EUPE | 0.163 | 0.090 |
| LeWorldModel | 0.034 | 0.022 |

Only DINO-WM shows meaningful action grounding. V-JEPA 2.1 and LeWorldModel's
action-conditioning is largely inert at this checkpoint — shuffling the input action barely
changes their predictions.

### Pixel-space reconstruction (one-step)

| | V-JEPA 2.1 | DINOv2 | EUPE | LeWorldModel |
|---|---|---|---|---|
| Ceiling LPIPS ↓ | **0.380** | 0.498 | 0.486 | 0.599 |
| LPIPS penalty | **+0.008** | +0.011 | +0.015 | +0.000 |
| Ceiling PSNR (dB) ↑ | **23.47** | 20.61 | 21.54 | 15.59 |
| Ceiling SSIM ↑ | **0.726** | 0.618 | 0.633 | 0.522 |

The ceiling row measures decoder quality alone: V-JEPA 2.1's decoder is sharpest;
LeWorldModel's — reconstructing a full frame from a single 384-d vector — is blurriest. With
decoder blur removed, one-step penalties are small across the board, consistent with the low
one-step NMSE values above.

### Summary

- **DINO-WM** is the only model that clearly grounds its predictions in the commanded action.
- **V-JEPA 2.1** reproduces the real trajectory's geometry best under rollout, despite the
  highest raw one-step error.
- **LeWorldModel's** very low raw error is an artefact of a near-static latent space, not
  evidence of superior prediction — confirmed independently by its own authors' ablations
  (SIGReg underperforms on low-diversity data) and by this project's rollout/action metrics.

## Repository Structure

```
worldmodelscope/
├── models/
│   ├── vjepa21/          # V-JEPA 2.1 encoder + custom action-conditioned predictor
│   ├── dinowm_dinov2/     # DINO-WM, original DINOv2 backbone
│   ├── dinowm_eupe/       # DINO-WM, EUPE backbone (this project's variant)
│   └── lewm/              # LeWorldModel
├── decoders/              # Per-model pixel decoders (training + frozen checkpoints)
├── metrics/               # NMSE, drift ratio, action reliance, LPIPS/PSNR/SSIM ceiling-penalty
├── data/                  # BridgeData V2 / Cosmos loading and preprocessing
├── configs/               # Per-model and per-experiment configuration files
├── scripts/
│   ├── train_decoder.py
│   ├── evaluate.py
│   └── plot_results.py
└── results/               # Generated tables and figures
```

## References

- LeCun, Y. *A Path Towards Autonomous Machine Intelligence.* OpenReview, 2022.
- He, K. et al. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022.
- Assran, M. et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR 2023.
- Assran, M. et al. *V-JEPA 2.* 2025. [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
- Mur-Labadia, L. et al. *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.* 2026. [arXiv:2603.14482](https://arxiv.org/abs/2603.14482)
- Zhou, G. et al. *DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning.* 2024. [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)
- Maes, L. et al. *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.* 2026. [arXiv:2603.19312](https://arxiv.org/abs/2603.19312)
- Walke, H. et al. *BridgeData V2: A Dataset for Robot Learning at Scale.* CoRL 2023. [arXiv:2308.12952](https://arxiv.org/abs/2308.12952)
- NVIDIA. *PhysicalAI WorldModel Synthetic Embodied Robot Scenes.* 2026.
