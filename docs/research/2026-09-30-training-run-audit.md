# Training run audit on September 30 2026

The latest run is safely stopped at Phase B epoch 173 of 1000, with 2730 of
11000 configured updates completed. Continuing its remaining 827 epochs under
the existing schedule is not a credible recovery strategy. The tumor Dice of
zero is a real prediction failure, accompanied by an effectively frozen
optimizer and defective spatial preprocessing. Fixing those engineering defects
does not establish reproduction of Alalwan's model or results.

## Evidence and scope

The audited source is revision `6b1e0e8`; the preserved run is
`models/overnight-initial-20260930-074744Z`. Its `stop-status.json`,
`checkpoints/training/events.jsonl`, and recovery/best checkpoints are the
run evidence. The terminal reason is `user stopped`, not a crash. No training
restart or parameter updates were performed during this audit. Four authorized
GPU forwards on representative cases supplement CPU checkpoint comparisons
and a target-label census. These forwards are diagnostic samples, not a new
full-cohort accuracy measurement.
Raw diagnostic scripts and JSON are preserved separately at
`models/audit-20260930/` in the original checkout. The CPU target census covers
all 28 training and 98 validation cases, before random training augmentation.

## Learning rate collapse

The optimizer correctly resets to 0.01 at the start of Phase B. StepLR then
halves it after each 10 logical epochs, each containing only 10 minibatch
updates. The checkpoint's LR is `7.62939453125e-8` after epoch 173. Phase B's
last epoch would use approximately `1.58e-32`, followed by a final scheduler
step to `7.89e-33`.

| Phase B epochs | LR used for updates |
| --- | --- |
| 1 to 10 | 0.01 |
| 91 to 100 | 0.00001953125 |
| 171 to 180 | 0.0000000762939453125 |
| 991 to 1000 | approximately 1.58e-32 |

Between recovery checkpoints for epochs 172 and 173, only 65,513 of 3,523,643
parameter elements changed (1.86%). The parameter delta had L2 norm `1.24e-7`,
relative norm `1.06e-9`, and maximum absolute change `7.45e-9`. BatchNorm
running statistics can still change even when learned weights barely move;
late validation variation is not evidence of substantial parameter learning.
Earlier Phase B updates did change the model, so this is not a claim that the
whole phase was wasted.

The paper's section 4.3 states this literal decay cadence. Section 3.2 also
states 100 and 1000 epochs with 10 steps or sub-epochs, without defining a step.
There is no demonstrated scheduler off-by-one or resume bug. Interpreting a
step as one minibatch gives little optimization before decay; that is an
unverified reconstruction choice. Even more updates per epoch would not remove
the literal schedule's vanishing endpoint. An alternative schedule would be an
explicit experimental interpretation, not a verified correction to the paper.

## Tumor failure and spatial preprocessing

Every one of the 173 Phase B validations reports zero tumor Dice. Four GPU
forwards using Phase A best, Phase B best, and latest weights on tumor-positive
training/validation examples also predict zero class-2 voxels. The sampled
training and validation targets contain 3695 and 5697 tumor voxels respectively.
The class-2 classifier gradient from weighted cross-entropy still points toward
raising tumor logits in these examples. Labels, loss class order, and metric
plumbing therefore do not explain away the failure as a display error.

The deterministic training-target census finds tumors in all 28 native masks,
but only 21 resized masks. Whole-volume resizing removes every tumor voxel in
seven cases. Tumor occupies only 0.04135% of resized training voxels; with the
configured class weights, its share of weighted target mass is about 0.4118%.
This mass is not a measured fraction of loss or gradient: individual voxel losses
also depend on predictions. It does explain why low average cross-entropy can
coexist with poor tumor predictions.
Validation loses all tumor labels in 15 of its 85 tumor-positive cases.

There is a separate confirmed coordinate bug. Intensities use trilinear
`align_corners=True`; labels use legacy `nearest`. At output depth index j,
the former samples `j*(D-1)/11`, while the latter selects `floor(j*D/12)`.
At native depth 861, the last image slice comes from index 860 but its mask
comes from index 789, a 71-slice displacement. Using the same output shape
did not guarantee anatomical alignment. The scale augmentation uses the same
inconsistent conventions in-plane.
The largest validation-volume displacement is 82 native slices.

The fix pairs trilinear `align_corners=False` with `nearest-exact` labels,
including augmentation and inference resizing. Both use the same half-pixel
grid; nearest labels round the image sampling position to a native voxel.
This follows the documented [PyTorch interpolation conventions](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html).
It fixes alignment, but does not make a 12-slice whole-volume representation
preserve small lesions. Their survival and learnability need separate validation.
The census under the corrected half-pixel sampling confirms that limitation:
only 19 of 28 tumor-positive training cases and 68 of 85 tumor-positive
validation cases retain any tumor label. Different sampled planes increase the
total tumor-voxel count while losing tumors in more individual cases. Therefore
the grid fix must not be mistaken for a complete data-representation fix.

## Paper reconstruction gaps

The reference is Alalwan et al., *Efficient 3D Deep Learning Model for Medical
Image Semantic Segmentation*, [DOI 10.1016/j.aej.2020.10.046](https://doi.org/10.1016/j.aej.2020.10.046).
The local PDF was checked directly. The paper contains contradictions and omits
details; none of the following ambiguities should be resolved silently.

- The narrative and Table 1 report 3.6 million parameters; Table 3 reports
  36,270,875 trainable and 36,433,587 total parameters for the DS-Conv model.
  Reducing block counts to match 3.6 million is not independently justified.
- Figure 1 has blocks `(4,12,24,36)`; this run used `(2,6,12,18)` and
  3,523,643 trainable parameters. The repository's full-block reconstruction
  has 10,795,323 under its other assumptions. Neither count proves the authors'
  actual architecture. Decoder DS-Conv is also a documented deviation from the
  figure's Conv3D labels.
- Section 3.2 describes successive training of the same model with best-weight
  transfer. It does not specify a liver-only first stage. This repository folds
  tumor labels into liver in Phase A, suppressing the tumor output before Phase
  B. Its contribution to collapse is plausible, not isolated by these diagnostics.
- Input size `224×224×12` is specified; construction of those inputs from full
  CT volumes is not sufficiently described. Whole-volume squeezing is an
  assumption, not a demonstrated match. Slabs or patches would be another
  interpretation requiring an explicit reconstruction decision and validation.
- The local training/validation cohort is not the paper's hidden-label LiTS
  leaderboard evaluation. Local Dice must not be presented as reproducing those
  reported scores.

## Corrections and next decision

Changes are isolated in the `fix/training-run-audit` worktree. The old run,
configuration, and checkpoints remain unchanged. Sampling grids are corrected,
and the preprocessing fingerprint prevents exact resume across the semantic
change. The prelaunch plan now exposes StepLR decay frequency and end rates;
epoch logs/events report LR used, next LR, and class Dice. Optimizer, loss,
phase targets, and model architecture are unchanged.

Do not resume this run expecting additional epochs to cure tumor collapse.
A new reproduction experiment needs an explicit choice of architecture evidence,
phase targets, input sampling, and scheduler interpretation. Before a long run,
require aligned image/label landmarks, retained tumor targets, and a short
tumor-positive overfit diagnostic demonstrating actual class-2 predictions and
improving tumor Dice. Those are acceptance gates, not a promise of the paper's
accuracy. This audit resolves the observed symptoms and fixes the proven grid
defect; it does not claim tumor accuracy has been repaired or the paper reproduced.

## Verification

The full CPU suite completed successfully: 291 passed and 3 skipped (294
collected). Ruff lint and formatting checks passed, as did the Git whitespace
check. Regression coverage includes coordinate landmarks, discrete labels,
inference consistency, rejection of old preprocessing fingerprints on resume,
the literal LR horizon, and LR event reporting across both phases. Independent
read-only review returned no surviving correctness findings.

The standard mypy invocation targeting Python 3.11 is blocked by Python 3.13
NumPy stubs using newer syntax in the installed environment. Mypy explicitly
targeting that installed Python 3.13 environment passes across all 29 source
files. The existing Python 3.11 virtual environment lacks project dependencies;
no local Python 3.11 type-check pass is claimed. After publication in PR 15,
the clean GitHub Actions Python 3.11 quality-gates job passed, including mypy.
Integration logs and
durable diagnostic evidence are in the separate audit artifact directory cited
above.
