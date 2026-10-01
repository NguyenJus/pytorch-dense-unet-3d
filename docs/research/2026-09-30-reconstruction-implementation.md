# Reconstruction implementation and gate ledger

Worktree: `/home/justin/projects/pytorch-dense-unet-3d-reconstruction`, branch
`feat/paper-reconstruction`, based on PR #15 head `9429a0e` including `230b761`.
PR #15 was verified open against `main`; the repository is public. The original
stopped run and all its checkpoints remain untouched. This is a new diagnostic
experiment, not a resumption or a paper-score reproduction.

## Current decision

The research and bounded implementation are reviewable. **The paper-reconstruction
objective is not complete and long training remains blocked.** Full-FOV in-plane
resize has already erased native tumor components in the all-case census. The
paper's exact topology, schedule and author framework semantics remain unresolved.
Passing software tests or a selected-sample overfit cannot remove those gates.

The selected diagnostic graph is `figure_skip_reconstruction_v1`, with
**64,591,723** parameters; it is not the earlier 54.7M profiling graph. It follows
full figure counts, standard decoder convolutions, the printed DB4 bottleneck32,
and the dashed skip sources. Explicit geometry repairs and framework assumptions
are recorded in the [topology evidence](2026-09-30-topology-evidence.md). The
[stage manifest](2026-09-30-model-manifest.json) records shapes, channels, skip
sources, exact counts and the complete model fingerprint. Historical reduced
weights remain evaluable through their known graph; model identity is never
inferred by fitting parameter count.

## R1–R9 results and downstream controls

| Gate | Disposition, evidence and checks | Consequence / issue |
| --- | --- | --- |
| R1 architecture | **Supported** full counts and standard decoder; **refuted** literal impossible padding/unresized skip geometry and historical half-count fidelity. **Provisionally adopted** stated-dimension repairs and resized figure skips. **Unresolved** exact skip taps, transition compression/order and contradictory channel/count claims. Figure was inspected visually; fixed-input CPU forward/backward and exact stage manifest passed. | Explicit diagnostic candidate only; no architecture lock. [#16](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/16). |
| R2 input construction | Input size is **supported**; author sample construction is **unresolved**. Native-depth slabs are a **provisional engineering representation** with executable coverage/coordinate checks. Full-FOV resize is **refuted as long-run input** by completely erased native 26-connected tumor components. Boundary-shell allowance was declared before census; no post-hoc tolerance change. | Stop this representation's training default. Native tiling investigated explicitly; production selection requires further integration/real-census/FOV-learning evidence. [#18](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/18). |
| R3 targets/transfer | Same three-class task and selected-A weight transfer **supported** by combined §§3–3.2; implicit liver-only phase is **refuted as paper fact**. Fresh optimizer/scheduler is **provisionally adopted**, author transfer-state details **unresolved**. CPU tests verify tumor gradients, best selection and exact resumed transfer. | `phase_a_targets=three_class`, `phase_transfer_policy=best_weights_fresh_optimizer`; no exact historical reset claim. [#17](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/17). |
| R4 schedule | Literal reporting is **supported**, epoch/subepoch/update meaning **unresolved**. The stopped run **refutes** the literal minibatch schedule as a useful long continuation. Figure2's 0–100 axes and DenseUNet157 legends were verified visually and retained as counterevidence. | Preserve A100/B1000 ×10 and halving/10 in unlaunchable reference config. Bounded short diagnostic has a separate identity; no floor/cosine/Adam substitution. [#17](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/17). |
| R5 loss | Eq.(2) weighted CE divided by voxel count is **supported** by independent visual transcription. Explicit valid-voxel mean, unchanged weights 0.2/1.2/2.2; padding absent from numerator/denominator. Manual value/gradient/padding tests pass. Summing unreduced CE avoids an unsupported strict CUDA reduced-NLL kernel without changing the equation. | `valid_voxel_mean` recorded in config/checkpoints. Historical weighted mean remains named and distinct. |
| R6 metric/reconstruction | Native whole-case scoring, liver union and strict tumor class **supported** by official LiTS sources. Historical pinned MedPy0.2.2 both-empty Dice0 **supported**; presence-aware exclusion **refuted as that evaluator's convention**. Uniform probability blending is **provisionally adopted** with explicit tests, not claimed author code. | Shared streaming predictor restores probabilities before argmax; padding excluded, complete voxel coverage required, interrupted/corrupt cohorts publish no score. Local holdout never equals hidden challenge evaluation. |
| R7 execution | Strict global CUDA determinism **refuted for this runtime/graph** by unsupported max-pool backward. Selected FP32/no TF32 mode explicitly disables strict global determinism while retaining deterministic cuDNN choice and benchmark off. Floating replay tolerance predeclared atol1e-6/rtol1e-4; integer/RNG/scheduler state must match exactly. | Bounded GPU diagnostics **failed** numerical replay. Training-mode memorization succeeded; eval-mode readiness is **inconclusive**, with its predeclared acceptance criterion unmet; [#19](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/19). No AMP, TF32, compile, fused optimizer or activation checkpointing was selected. GTX1080/Pascal/8GB are not gates. |
| R8 augmentation/sampling details | HU clamp, random scale range and mirror are **supported**; exact axes, probabilities, interpolation, padding, order and author sampling **unresolved**. Diagnostic augmentation is explicitly disabled, a named deviation. Slab stride/tail/validity/blending choices are **provisional engineering choices**, proven for coverage and coordinate consistency only. | No claim of verified augmentation or stochastic-retention pass. Production augmentation remains in [#18](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/18); never use labels for inference sampling. |
| R9 framework | Versioned framework defaults support explicit provisional BN epsilon/momentum and initialization choices, but exact author versions/overrides **unresolved**. Equivalence of Keras LR-scaled velocity and PyTorch gradient-buffer SGD at decay **refuted analytically**. BN running-variance conventions also differ. | Explicit graph/optimizer identities and tests; no cross-framework numerical-equivalence claim. [#16](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/16), [#17](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/17). |

Each detailed assumption, positive support, counterevidence and source location
is in the [topology](2026-09-30-topology-evidence.md),
[recipe](2026-09-30-recipe-evidence.md) and
[spatial](2026-09-30-spatial-evidence.md) evidence records. A bounded search found
no verified author implementation; absence is explicitly unresolved, not proof.

## Implemented experiment contract

- `configs/historical-reference.yaml`: named historical graph, weighted mean and
  liver-only A for comparison. The stopped run is neither modified nor resumed.
- `dense_unet_3d/config.yaml` and `configs/reconstruction-reference.yaml`: literal paper schedule plus explicit
  diagnostic graph/data/numerical interpretations; `mode: reference` refuses launch.
- `configs/reconstruction-diagnostic.yaml`: A1/B1 ×10 updates, microbatch/effective
  batch1, no accumulation, finite 900-second invocation/cumulative allocation.
  This is a short engineering protocol, not replacement paper epoch semantics.
- [Split manifest](2026-09-30-split-manifest.json): original 28 training and 98
  validation IDs, enforced before dataset construction. The native index contains
  1,257 training and 3,588 validation slabs. Native tiles would require 9× as
  many samples, and are an unselected follow-up.
- Source files are hashed once per invocation, independently of repeated slabs;
  geometry, sample index, padding, labels, loss, optimizer, schedule, execution
  and model graph enter continuation identity. Schema2 rejects old exact-resume
  state. Best/last/recovery checkpoints carry graph and experiment metadata;
  top-level and selected-best metadata are validated before resume/export.
- Zero-worker standard loaders retain default collation and RNG restoration.
  Native tuple/dict inputs normalize to class indices with padding `-100`.
  CPU interrupted/uninterrupted weights, optimizer, scheduler, RNG and phase
  transfer agree exactly, including native sample order and padded depths.
- Validation is whole-case and target-independent. The CLI uses the same predictor;
  `eval --max-cases` aliases the existing bound (cases for native sampling,
  batches for historical sampling). A partial cohort is withheld. Budget/stop
  during validation retains the previous complete recovery transaction.
- Events record actual sample/update exposure, target/predicted class counts,
  positive samples, class-weighted CE sums, LR, full gradient norm and a declared
  bounded parameter-update sample. Positive epochs with zero tumor predictions
  are flagged; negative-only batches are not failures. No diagnostic silently
  changes LR, loss weights or stopping policy.

## Execution and verification artifacts

[Runtime receipt](2026-09-30-reconstruction-runtime.json) pins the inspected
Python3.13.13 / torch2.13.0+cu130 environment and RTX5070Ti. GPU diagnostics use
at most100 total updates and900 seconds across attempts; old runs are untouched.
[GPU diagnostic receipt](2026-09-30-gpu-diagnostics.md) includes cold/warm memory,
finite updates, BN behavior, numerical replay and real tumor-positive samples.

The correct reconstruction checkout passed **360 CPU tests, 3 skipped**.
After final loss/execution/type fixes, **42 focused tests passed**; mypy targeting
the installed Python3.13 passed across34 source files. Independent strong-model
review found and resolved inference execution/metadata and unmanaged-launch
bypasses. A verifier's earlier results from the original checkout were discarded;
they are not evidence for this branch. Final Ruff lint/format and whitespace checks passed; expanded mypy passed
all 37 package/audit-script files. Final targeted checks additionally passed 11 slab
tests, 15 reference/config tests and 45 phase/native-recovery tests (overlapping
checks are not added to the full-suite count). Diagnostic outcomes follow below.
Raw correct-checkout logs remain local-only under
`models/reconstruction-20260930/verification/`; the reported exit status is not a
claim that those logs ship in a clean clone.

The repository's default Python3.11 mypy target cannot parse installed
Python3.13 NumPy stubs; no Python3.11 local type-check pass is claimed. This is
an environment compatibility limit, not a reason to downgrade the modern GPU
runtime. Use the recorded supported-runtime command in training operations.


## Final retention and GPU outcomes

The [complete census summary](2026-09-30-retention-census-summary.json) records
126 cases, **869 native components, 22 completely erased**: training 4/138 across
3 cases; validation 18/731 across 7 cases. No depth slice was omitted; every
predeclared core/boundary test passed. Those facts do not override the independent
zero-erasure requirement. Census exit2 is the expected completed negative gate.
Raw component identities and per-component physical-volume results are retained
at `models/reconstruction-20260930/retention-census.json` with its SHA256 in the
tracked summary. A header-only follow-up found 106 mm declarations and 20 missing
spatial-unit declarations. Millimeters for the latter remain a named provisional
LiTS-provenance inference: per-case retention ratios and zero-erasure results do
not depend on that inference, but absolute/pooled physical-volume figures do.
The census utility now converts declared meter/micron units and labels unknown
units explicitly; no current cohort scalar changes because all declared units
are mm. Native tiling feasibility passed six synthetic geometries and an oblique
landmark, but production integration and real-label evidence remain [#18](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/18).

GPU work consumed **90 optimizer updates and 301.525 seconds** of the declared
100-update/900-second allocation, with no long launch. Warm forward peak allocated
memory was **2.397 GiB**. Finite gradients/updates, once-per-step BN state updates,
and eval-state preservation passed. The numerical replay **failed** the declared
floating tolerance, while RNG/scheduler/integer state matched exactly. Divergence
started before the stop boundary, so the evidence does not isolate a serialization
bug. See the receipt for full output/loss/gradient comparisons.

The separate 80-update training-sample overfit achieved approximately 0.869 mean
train-mode tumor Dice near its end. **Eval-mode mean tumor Dice was 0.02145→0**,
despite loss 6.0435→0.32115. Final class2 predicted counts 740/766 had zero overlap
with true tumors. This demonstrates training-mode memorization. Evaluation readiness remains
inconclusive after 80 updates: the small near-zero Dice decline alone does not
establish degradation or model failure. The predeclared eval acceptance criterion
was unmet; its recorded flag remains false. No BN recalibration, optimizer sweep, tolerance
relaxation or alternative weights were used. The final overfit weights were not
saved before process exit, so a same-final-weight batch-stat/running-stat causal
comparison is **untested**; the runner now saves that checkpoint for a future
allocation. Exact source for both update-bearing attempts and their config is in
the tracked [artifact archive](artifacts/README.md); the zero-update and
reporting-only source is retained by hash only, and large raw
tensors/checkpoints remain local-only.
This limitation, failed replay, and inconclusive evaluation readiness are tracked in [#19](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/19).

Readiness remains blocked by [#16](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/16),
[#17](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/17),
[#18](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/18), and
[#19](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/19). These are
required tracked follow-ups, not an implicit authorization to start long training.


Next action: use issues #16–#19 to resolve the exact remaining gates. The highest
value engineering work is numerical repeatability/fixed-weight normalization
isolation and production native-tile integration, with independent evidence
checks. Do not start a long run or relabel an alternative optimizer/schedule as
paper reconstruction. Code remains in the isolated task worktree for review;
no PR merge or training continuation was performed.
