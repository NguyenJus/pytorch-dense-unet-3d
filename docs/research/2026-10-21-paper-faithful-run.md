# Paper-faithful training run: evidence ledger (2026-10-21)

Source design: `docs/specs/2026-10-21-paper-faithful-training-run-design.md` (local, gitignored). Section 2 below is copied verbatim from that design and is binding. Launchable configs: `configs/reconstruction-run.yaml` (full run) and `configs/reconstruction-smoke.yaml` (bounded smoke). Their `experiment.fidelity` map records one `{label, evidence}` entry per row group and is validated by `training/experiment.py::validate_experiment` in `mode: training`.

| Label | Meaning |
| --- | --- |
| **LITERAL** | The paper states it; copy it exactly. |
| **INFERRED** | The paper omits it; one choice is made, with the evidence used (paper-internal arithmetic, Fig. 1/2, or the H-DenseUNet lineage the paper copies from). |
| **DEVIATION** | The paper states it but the literal reading cannot work; the replacement is the nearest working reading and the reason is recorded. |

The project's working hypothesis (owner, 2026-10-21): the paper is not fully
trustworthy. A **DEVIATION** is therefore not a failure of fidelity; an
**unlabelled** deviation is.

## 1. Audit findings: why the previous reconstruction config could not succeed

| # | Finding | Evidence |
| --- | --- | --- |
| 1 | "10 steps per epoch" read as 10 minibatch updates gives 11,000 updates at batch 1, under one pass over 11,313 training tiles, with LR halving every 100 updates (1e-5 after Phase A; 1.6e-32 at the end of Phase B). | `2026-09-30-training-run-audit.md`; stopped run frozen by Phase B epoch 173. |
| 2 | Split is 28 train / 98 validation; the paper trains on all 131. | `2026-09-30-split-manifest.json`. |
| 3 | Validation every epoch over 98 whole cases with native tiles: 32,292 forwards × 0.147 s ≥ 79 min per validation, ~1,450 h over 1,100 epochs. | Warm forward timing in `2026-09-30-gpu-diagnostics.md`; 32,292 from `-spatial-evidence.md`. |
| 4 | Uniform deterministic tile sampling: of 11,313 training tiles 16.1 % contain liver, 4.2 % contain tumor; tumor is 0.064 % of native voxels. Augmentation is hard-rejected for native slabs. | Census run 2026-10-21 (`tile_bounds` over the 28 training masks). |
| 5 | No launchable config exists: reference mode refuses launch, diagnostic mode caps at 20 updates, execution contract forbids TF32/cuDNN benchmark/workers, exact resume forbids `num_workers>0` and non-standard samplers. | `training/experiment.py::validate_experiment`, `training/recovery.py::_loader_identity`. |

## 2. Fidelity ledger (the design, piece by piece)

Each row is binding. Implementers copy the "Decision" column; S6 copies the
whole table into the research ledger.

### 2.1 Architecture

| Piece | Paper says (where) | Decision | Label |
| --- | --- | --- | --- |
| Input | 224×224×12, 1 channel (§3, Eq. 1, Fig. 1) | `(N,1,12,224,224)` NCDHW | LITERAL |
| Output | 3-class per-voxel softmax (Eq. 1) | logits `(N,3,12,224,224)`; softmax inside loss/predictor | LITERAL |
| Block counts | 4×, 12×, 24×, 36× (Fig. 1) | `(4,12,24,36)` | LITERAL |
| Growth, compression | growth 32 (Fig. 1); compression 0.5 (§3) | 32 / 0.5 | LITERAL |
| Dense layer | Conv1 → BN → ReLU → DS-Conv3 → BN → ReLU; bottleneck 128 (DB1–3), 32 (DB4) as printed | as implemented in `model/reconstruction.py::FigureDenseLayer` with `(128,128,128,32)` | LITERAL (printed), provenance unresolved |
| Stem / pool | Conv k7 s2 → 96; MaxPool k3 s2 (Fig. 1); printed sizes 112×112×6, 56×56×3 | SAME-upper padding so printed sizes hold | DEVIATION (literal pad=0 cannot produce the printed sizes) |
| Transitions | Conv1 → BN → ReLU → Conv1 stride 2 (Fig. 1); depth stays 3 | stride `(1,2,2)` | LITERAL order; stride INFERRED from printed depth |
| Decoder | 5 upsampling layers, bilinear ×2, Conv3D k3 → 504/224/192/96/64, BN, ReLU (Fig. 1, §3) | `FigureUpsample` as implemented; trilinear is the 3-D reading of "bilinear" | LITERAL widths/op; interpolation INFERRED |
| Skips | dashed arrows stem→up5, DB1→up4, DB2→up3, DB3→up2, DB4→up1 (Fig. 1 at 300 dpi, verified 2026-10-21) | keep `figure_skip_reconstruction_v1` exactly as drawn, including the DB4 self-concat at up1 (64,591,723 params) | LITERAL as drawn; the self-concat is flagged as a probable drawing artifact in the ledger, **not** changed in this spec |
| BN | not stated; TF/Keras (§4.3) | eps 1e-3, momentum 0.01 (Keras-era defaults), as implemented | INFERRED (framework) |
| Init / bias | not stated | Glorot uniform, zero bias, all convs biased, as implemented | INFERRED (framework) |

No model code changes are required. `model/config.py` identity `figure_skip_reconstruction_v1` is reused unchanged so existing fingerprints, manifests and tests stay valid.

### 2.2 Data and sampling

| Piece | Paper says | Decision | Label |
| --- | --- | --- | --- |
| Dataset | LiTS-2017, 131 labeled training volumes, 70 hidden test (§4.1) | all 131 local cases; new split **105 train / 26 validation** (open question Q2) by stratified random draw: cases sorted by tumor voxel count, every 5th into validation, seed 20261021, written to `docs/research/2026-10-21-split-manifest.json` with per-case tumor presence | INFERRED (no hidden test set available) |
| HU window | truncate to [−200, 250] (§4.1) | clamp, as implemented | LITERAL |
| Training sample | "224×224×12" (§3); construction not described | **Foreground-centred random crop**, native in-plane spacing, 12 contiguous native slices: centre voxel drawn from the case's tumor voxels with p = 0.5 when the case has tumor, otherwise from liver voxels; crop extent in-plane = round(224·s) with s ~ U(0.8, 1.2) (see augmentation), then resized to 224×224 (trilinear image, nearest-exact mask, half-pixel grid); depth never resampled; out-of-volume region padded with HU −200 / target −100 | INFERRED from the H-DenseUNet lineage (same 0.8–1.2 scale, same HU window, same decoder widths; its generator uses liver/tumor-centred crops 50/50) |
| Samples per step | "each epoch contains 10 steps (sub-epochs)" (§3.2) | **one step = one pass over the training cases, one crop per case**; one epoch = 10 passes = 10·N_train samples (1,050 at N_train = 105) | DEVIATION from the repo's prior "step = minibatch" reading; evidence: Table 2 reports per-epoch time measured over 5/10/15/20 *cases*; §4.3's 42 h at ~0.1–0.2 s per sample on a GTX 1080 implies 10⁵–10⁶ presentations, not 11,000 |
| Mirror | "random scaling and mirror from 0.8 to 1.2" (§4.1) | flip along tensor W with p = 0.5 (open question Q4) | LITERAL (mirror) / INFERRED (axis, probability) |
| Scaling | same sentence | in-plane only, uniform factor, applied as crop-size scaling before resize (above) | LITERAL (range) / INFERRED (mechanism, from H-DenseUNet) |
| Validation / inference input | not described | existing `native_tiles_v1` deterministic grid (depth stride 12 + tail, in-plane 224 tiles at starts 0/224/288), uniform probability averaging over overlap, whole-case argmax on the native grid, labels never used to choose windows | INFERRED (engineering; already verified zero lesion erasure) |
| Batch size | not stated | microbatch 2 (open question Q3) | INFERRED |

### 2.3 Loss, optimizer, schedule, cascade

| Piece | Paper says | Decision | Label |
| --- | --- | --- | --- |
| Loss | Eq. (2): −(1/N) Σ_i Σ_c w_c y_ic log ŷ_ic; weights 0.2 / 1.2 / 2.2 (§3.1) | `loss_reduction: valid_voxel_mean`, padding excluded, as implemented | LITERAL |
| Optimizer | momentum 0.5, initial LR 0.01, "decreased by a value of 2 after every 10 epochs" (§4.3) | SGD lr 0.01, momentum 0.5, dampening 0, no Nesterov, no weight decay; StepLR step 10 epochs, gamma 0.5; **one scheduler step per epoch as defined in §2.2** | LITERAL (values) / INFERRED (plain SGD, PyTorch buffer semantics) |
| Phase A | 100 epochs × 10 steps (§3.2) | 100 epochs; LR 0.01 → 1.95e-5 at epoch 100; ~5,250 updates per plateau at batch 2 | LITERAL |
| Transfer | "optimal weights obtained in the initial phase are used as a base" (§3.2) | load Phase A best weights; fresh optimizer and scheduler (LR back to 0.01) | LITERAL (weights) / INFERRED (reset) |
| Phase B | 1000 epochs × 10 steps (§3.2) | **100 epochs** (open question Q1) | DEVIATION: with halving every 10 epochs the literal 1000 epochs spend ~800 epochs below LR 1e-8; Fig. 2 plots the "2nd training phase" over 0–100 epochs |
| Phase targets | same model, same 3-class task both phases (§3, §3.2) | three_class in both phases | LITERAL |
| Best selection | "optimal weights" undefined | mean of liver and tumor per-case Dice on the validation split (existing `foreground_selection_score`); evaluated every 10 epochs and at the final epoch of each phase | INFERRED |
| Metric | Dice per case and global (§4.2, LiTS) | existing whole-case native evaluation with liver = classes ≥1, tumor = class 2, empty-denominator → 0 (MedPy 0.2.2 convention) | LITERAL |

### 2.4 Execution (no paper claim; engineering)

| Setting | Decision | Rationale |
| --- | --- | --- |
| Precision | FP32 weights/activations with TF32 matmul/conv **enabled**; bf16 autocast available as an explicit `execution.autocast: bf16` opt-in, default off | TF32 is numerically far above the Keras/TF-1 noise floor already accepted; autocast kept opt-in for Q3 |
| cuDNN | `benchmark: true`, `deterministic: false` | strict determinism is refuted and a non-goal |
| Loader | `num_workers: 4`, `pin_memory: true`, persistent workers | GPU-bound training; crops are small partial reads of uncompressed `.nii` |
| Validation cadence | `validation_every: 10` plus phase-final epoch; whole-case native tiles on the 26-case holdout, tiles batched (batch 8) in the predictor, each case's image loaded once | ~9,400 tile forwards per validation ≈ 10 min batched; 22 validations ≈ 4 h total |
| Recovery | epoch-boundary recovery checkpoint every epoch (unchanged); RNG restored at the boundary; **continuation is epoch-exact, not sample-order-bitwise** when workers > 0 | see §3.4 |
| Budget | `--wall-seconds` per invocation, `--budget-seconds 172800` (48 h) persistent | expected ~20–24 h training + ~4 h validation at batch 2 TF32; see §5 |

Expected update counts (N_train = 105, batch 2): 525 updates/epoch, 52,500 per phase, 105,000 total; 210,000 sample presentations.

**As-run amendment (2026-10-21, owner decision).** The tables above are the design as specified. Preflight rejected LiTS cases 48-52: their segmentations carry an identity affine while the volumes carry the scanner geometry, so the image/mask grid check fails. The 2026-09-30 split had already held only the other 126 cases. The owner chose to exclude the five rather than repair their headers; `make_split_manifest.py` records them under `excluded` with the reason and hashes. The as-run split is **101 train / 25 validation**, and the counts become 505 updates/epoch, 50,500 per phase, 101,000 total; 202,000 sample presentations.

**As-run amendment (2026-10-21, owner decision): cuDNN benchmark off, inference batch 4.** The owner's rule is "whichever fits on gpu, alone, without any other leakage". The first GPU smoke failed in its first validation with `CUDA error: unknown error`, and the WSL driver logged `dxgkio_make_resident: Ioctl failed: -12` (out of memory). Measured on the RTX 5070 Ti (16 GiB), eval/no-grad, 12×224×224 tiles, TF32 on: without cuDNN benchmark the isolated forward peak is 1.76 / 3.27 / 6.30 / 12.35 GiB at batch 1 / 2 / 4 / 8, about 1.5 GiB per tile. With benchmark the warm forward costs 0.035 s/tile at batch 2, 0.033 at batch 4 and 0.040 at batch 8. Batch 8 cannot fit next to the training state, so the smoke ran with benchmark on and `dataset.inference_batch` 2 (spec §5 OOM fallback). A follow-up measurement on the same card (Windows holds ~1.3-1.6 GiB; training batch 2; deepest validation case 512×512×987) found that benchmark on leaks: its algorithm search makes training alone reserve 13.67 GiB, and device free memory reaches 0.00 GiB at every inference batch, spilling into WSL shared memory. With benchmark off, training costs 0.126 s/sample at a 7.38 GiB peak reserved, and the deepest case evaluates in 44.8 / 37.5 / 34.7 / 166.9 s at inference batch 1 / 2 / 4 / 6, peak reserved 7.38 / 7.38 / 10.83 / 16.00 GiB, minimum device free 7.22 / 7.22 / 3.77 / 0.00 GiB. Batch 6 spills. Both configs therefore set `execution.cudnn_benchmark: false` and `dataset.inference_batch: 4`, the fastest leak-free choice with 3.77 GiB headroom; the predictor default is 4 as well. This overrides the cuDNN row and the batch-8 validation note in §2.4. It is engineering, not fidelity; predictions are array-equal across batch sizes on CPU.

## 3. Plan-level decisions

These resolve places where the design was silent or self-inconsistent; the owner's delegate accepted them on 2026-10-21.

- **D1 Split membership.** `make_split_manifest.py --link-root DIR` also writes symlink trees `DIR/train/` and `DIR/validation/` (original file basenames kept). The config points `train_img_dirs`/`test_img_dirs` at those trees, so `_validate_split_manifest` (discovered cases must equal the manifest set) and the overlap checks stay unchanged.
- **D2 Inference representation key.** The design's `inference_representation` is the existing `dataset.inplane_representation: native_tiles_v1`. The new key is `dataset.training_representation: foreground_crops_v1`.
- **D3 Crop randomness transport.** A `CropSampler` in the main process yields `(index, item_seed)` tuples from a `torch.Generator`; `__getitem__` derives centre, scale and flip only from `item_seed`. The generator is captured by the recovery RNG machinery, so a restored boundary replays crop parameters exactly for any worker count. New key: `dataset.sampling_seed`.
- **D4 `training_pass` arithmetic.** `updates_per_epoch = ceil(passes_per_epoch * N_train / batch_size)` = 525 updates/epoch at 105 cases, batch 2 (equals `len(loader)`); 105,000 total over both phases. Each `training.phase_{a,b}_steps_per_epoch` must equal `dataset.passes_per_epoch`.
- **D5 One training loader for both phases.** Both phases are three_class, so Phase B reuses the Phase A train loader object: one generator stream (Phase B does not replay Phase A's crops) and one foreground index.
- **D6 Execution keys.** A present execution block always needs exactly the five keys `precision`, `tf32`, `deterministic`, `cudnn_benchmark`, `autocast`; nothing is defaulted. The plan's 3-key `{precision, tf32, deterministic}` form was removed by owner decision 1 (2026-10-21, no backward compatibility), so a 3-key block is now rejected. `mode: diagnostic` keeps the strict FP32 reference (`tf32: false`, `cudnn_benchmark: false`, `autocast: none`). `mode: training` requires all five keys explicitly and allows `tf32: true`, `cudnn_benchmark: true`, `autocast: none|bf16`.
- **D7 Loader workers** apply only to the training loader. Validation never iterates its loader (`evaluate` reads `dataset.cases`).

### As-built notes

- Foreground sampling uses liver = labels >= 1 (whole liver including tumor, the evaluation convention) and tumor = label 2, so a liver-centred draw can land on a tumor voxel.
- `fidelity` has 18 entries: the 17 required keys plus `decoder_up1_self_concat` (LITERAL as drawn, flagged as a probable drawing artifact).
- The recovery loader identity always records the `workers` mapping (`num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`), single-process loaders included. Per owner decision 1, identities recorded before this change no longer match, so older run directories cannot be resumed.

## 4. Alternatives considered

- **Keep "step = minibatch" and raise steps_per_epoch to ~525.** Equivalent
  update count, but hides the pass structure the paper's Table 2 implies and
  makes the per-case exposure accidental. Rejected.
- **Phase B = 1000 epochs with an LR floor.** Adds an unstated floor; worse
  fidelity than reading Fig. 2's 100-epoch axis. Rejected (Q1 lets the owner
  override).
- **Uniform random tiles instead of foreground-centred crops.** 4.2 % tumor
  tiles; would spend ~96 % of updates on background. Rejected on the audit
  census.
- **Full-FOV 224×224 resize of whole slices.** Refuted by the 22 erased tumor
  components (`2026-09-30-retention-census-summary.json`). Rejected.
- **Drop the DB4 self-concat at up1 (saves 22.6 M params).** Probably the
  authors' real graph, but not what the figure draws; deferred to a named
  follow-up variant unless memory/throughput force it.
- **Train/validation 118/13.** Closer to "all 131" but tumor Dice on 13 cases
  is too noisy for checkpoint selection. Offered as Q2.

## Config fidelity map

The `experiment.fidelity` block of `configs/reconstruction-run.yaml` (and the smoke config), as committed. It is recorded in every checkpoint's experiment metadata and printed by the training plan.

| `experiment.fidelity` key | Label | Evidence recorded in the config |
| --- | --- | --- |
| `stem_pool_padding` | DEVIATION | Fig. 1 stem Conv k7 s2 -> 96 and MaxPool k3 s2 print 112x112x6 and 56x56x3; literal pad=0 cannot produce the printed sizes, so SAME-upper padding is used. |
| `transition_stride` | INFERRED | Fig. 1 transition order Conv1 -> BN -> ReLU -> Conv1 stride 2 is LITERAL; the printed depth stays 3, so the stride is (1,2,2). |
| `decoder_interpolation` | INFERRED | Fig. 1 and section 3 state bilinear x2 upsampling with Conv3D k3 widths 504/224/192/96/64 (LITERAL); trilinear is the 3-D reading of bilinear. |
| `decoder_up1_self_concat` | LITERAL | Fig. 1 at 300 dpi draws DB4 -> up1; figure_skip_reconstruction_v1 keeps the DB4 self-concat at up1 as drawn (64,591,723 params), flagged as a probable drawing artifact and not changed. |
| `batch_norm_defaults` | INFERRED | BN settings are not stated; TF/Keras framework (section 4.3) gives Keras-era defaults eps 1e-3, momentum 0.01. |
| `initialization` | INFERRED | Initialization and bias are not stated; Keras framework defaults: Glorot uniform, zero bias, all convolutions biased. |
| `split` | INFERRED | Paper trains on LiTS-2017's 131 labeled volumes with a hidden 70-case test set (section 4.1); no hidden test set is available and cases 48-52 are excluded (their segmentations carry an identity affine that fails the image/mask grid check), so 101 train / 25 validation of the 126 valid cases, stratified by tumor voxel count (every 5th into validation), seed 20261021, recorded in docs/research/2026-10-21-split-manifest.json. |
| `training_sample` | INFERRED | Section 3 gives 224x224x12 but not its construction; foreground-centred random crop (tumor centre p=0.5 when present, else liver), native in-plane spacing, 12 contiguous native slices, from the H-DenseUNet lineage (same 0.8-1.2 scale, HU window, decoder widths; liver/tumor-centred crops 50/50). |
| `step_unit` | DEVIATION | Section 3.2 'each epoch contains 10 steps (sub-epochs)' is read as one step = one pass over the training cases (one crop per case), not one minibatch: Table 2 reports per-epoch time over 5/10/15/20 cases and section 4.3's 42 h at ~0.1-0.2 s/sample implies 1e5-1e6 presentations, not 11,000. |
| `mirror` | INFERRED | Section 4.1 'random scaling and mirror' is LITERAL; axis and probability are not stated, so flip along tensor W with p=0.5. |
| `scaling` | INFERRED | Section 4.1 range 0.8-1.2 is LITERAL; mechanism inferred from H-DenseUNet: uniform in-plane factor applied as crop-size scaling before the resize to 224x224. |
| `inference_tiling` | INFERRED | Validation/inference input is not described; existing native_tiles_v1 deterministic grid (depth stride 12 + tail, in-plane 224 tiles), uniform probability averaging over overlap, whole-case argmax; verified zero lesion erasure. |
| `batch_size` | INFERRED | Batch size is not stated; microbatch 2. |
| `optimizer_semantics` | INFERRED | Section 4.3 momentum 0.5, initial LR 0.01, halved every 10 epochs is LITERAL; plain SGD (dampening 0, no Nesterov, no weight decay) with PyTorch gradient-buffer semantics and one StepLR step per epoch is inferred. |
| `phase_transfer_reset` | INFERRED | Section 3.2 'optimal weights obtained in the initial phase are used as a base' is LITERAL for the weights; a fresh optimizer and scheduler (LR back to 0.01) is inferred. |
| `phase_b_epochs` | DEVIATION | Section 3.2 states 1000 epochs x 10 steps; with halving every 10 epochs the literal 1000 epochs spend ~800 epochs below LR 1e-8, and Fig. 2 plots the 2nd training phase over 0-100 epochs, so Phase B runs 100 epochs. |
| `best_selection` | INFERRED | 'Optimal weights' is undefined; best = mean of liver and tumor per-case Dice on the validation split (foreground_selection_score), evaluated every 10 epochs and at each phase-final epoch. |
| `execution` | DEVIATION | No paper claim (engineering): FP32 weights/activations with TF32 matmul/conv enabled, cuDNN benchmark off (its algorithm search exhausts the 16 GiB card and spills into shared memory), which makes cuDNN select deterministic algorithms (`torch.backends.cudnn.deterministic = not cudnn_benchmark`), while `torch.use_deterministic_algorithms` stays off, bf16 autocast opt-in only; departs from the paper-era FP32 TF/Keras execution and is recorded as engineering, not fidelity. |

## Results

The full-run lines are filled after the run.

- Split manifest (`docs/research/2026-10-21-split-manifest.json`): 101 train / 25 validation of 126 valid cases (48-52 excluded); 90 and 23 cases with tumor. `preflight --full-decode` passes.
- Bounded smoke (`configs/reconstruction-smoke.yaml`, RTX 5070 Ti, FP32 + TF32, cuDNN benchmark, batch 2, 4 workers, no other GPU job), 2026-10-21. Measured history: this smoke ran with cuDNN benchmark on and inference batch 2, before the configs moved to benchmark off and inference batch 4 (see the §2.4 amendment).
  - Startup (preflight full decode + foreground index build): ~740 s per invocation.
  - Training: 318 / 307 / 301 s per 1,010-sample epoch = **0.30-0.32 s/sample**, below the 0.5 s/sample bf16 threshold, so autocast stays off.
  - Validation (25 cases, inference batch 2): 478 / 439 / 431 s.
  - Loss 0.207 → 0.177 (Phase A epochs 1-2) → 0.169 (Phase B epoch 1); per-case Dice liver 0.24 → 0.37 → 0.43, tumor 0.014 → 0.030 → 0.036; 16.6 M class-2 voxels predicted in Phase A epoch 1. All losses finite.
  - Stop/resume: stopped at the epoch-0 boundary, then again during Phase A epoch 2 validation. The interrupted validation published nothing and epoch 2 replayed from the epoch-1 boundary. Resumed to completion; 4,266 s cumulative of the 5,400 s budget.
  - GPU batch 1 vs batch 2 labels on validation case 3 (512×512×534): 4,685 of 139,984,896 voxels differ (0.0033 %, limit 0.01 %).
  - Projected full run: 200 × ~305 s training + 20 validations × ~440 s + startup ≈ 19.6 h, inside the 48 h budget and likely inside one 24 h invocation. These smoke timings ran with cuDNN benchmark on and inference batch 2. Training is input-bound: the smoke ran 0.30-0.32 s/sample, while the model step alone takes 0.121 s/sample with benchmark on and 0.126 with it off. Validation of the deepest case takes 34.7 s at batch 4 with benchmark off and 37.2 s at batch 2 with benchmark on. The projection should therefore hold for the as-run settings; the first epochs of the full run will confirm it.
- Full run: validation curve (liver and tumor Dice per case, every 10 epochs), best epoch and score: TBD
- Tumor Dice nonzero on the 25-case holdout: TBD
- Local holdout numbers are not comparable to the paper's hidden 70-volume test set.
