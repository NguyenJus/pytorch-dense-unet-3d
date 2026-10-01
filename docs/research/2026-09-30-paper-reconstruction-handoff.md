# Alalwan reconstruction implementation handoff

Implement a defensible reconstruction of Alalwan's liver and tumor segmentation
architecture and training recipe. Paper fidelity is the primary objective;
GTX 1080 compatibility and an 8 GB memory limit are not requirements. Modern
Blackwell hardware, newer compute capabilities, and execution optimizations are
allowed, with their numerical effects validated and recorded. Preserve the
strongest paper evidence, verify weaker assumptions, and make contradictions explicit. This is
an implementation spec with research gates, not a claim that the paper is fully
specified or that its scores can already be reproduced.

This scope supersedes earlier hardware-certification requirements. The paper's
GTX 1080 claim remains contextual evidence for judging reconstruction plausibility,
not a deployment target, completion gate, or reason to require a Pascal environment.

## Starting state and scope

- Start from [PR 15](https://github.com/NguyenJus/pytorch-dense-unet-3d/pull/15),
  branch `fix/training-run-audit`, including fix commit `230b761`. Confirm the PR
  state and base before branching; do not implement on an older uncorrected tree.
- Local checkout: `/home/justin/projects/pytorch-dense-unet-3d-audit`.
  The original checkout contains the stopped run and ignored diagnostic artifacts.
- Read [the run audit](2026-09-30-training-run-audit.md) and the older architecture
  decision record. Treat old decisions as historical assumptions, not authority.
- Preserve `models/overnight-initial-20260930-074744Z` and its recovery points.
  It stopped at B173 with tumor Dice zero and negligible parameter updates.
  This work creates a new experiment identity; do not modify checkpoints to
  bypass resume checks or restart the old run.
- Already fixed: image/mask half-pixel alignment, preprocessing identity rejection
  across that change, LR horizon warnings, and per-epoch LR/Dice visibility.
- Implement research, configuration, model/data/evaluation changes and CPU tests.
  A long training launch is not part of this handoff. Obtain an explicit bounded
  GPU allocation for implementation diagnostics or training when executing it;
  this session's authorization covered the profiles used to write this spec.

## Evidence and decision rules

Primary reference: Alalwan et al., *Efficient 3D Deep Learning Model for Medical
Image Semantic Segmentation*, [DOI](https://doi.org/10.1016/j.aej.2020.10.046),
published pages 1231–1239. The locally available PDF is
`docs/papers/alalwan2020.pdf` in the original checkout; it is not tracked. Use
the published diagram/equations, not text extraction alone, for topology and loss.

Evidence priority is direct implementation evidence from a verified author source,
then mutually consistent paper text/figure/equations, then explicitly identified
reconstruction inference supported by executable checks. Author code is not
automatically decisive: verify provenance and agreement with the described model.
Related papers provide context, not proof of this paper's implementation.

The text/Table 1 say 3.6 million parameters; Table 3 says 36,270,875 trainable
and 36,433,587 total. Do not tune width/depth to a parameter-count target or reuse
the repository's arbitrary ±15% band as a fidelity acceptance test. Likewise,
do not fit the advertised 569-layer label by inventing an operation-count rule.

Every assumption below must receive a recorded disposition: **supported**,
**provisionally adopted**, **refuted**, or **unresolved**. Record source location,
interpretation, counterevidence, checks, and which downstream tasks it controls.
Provisionally adopting a choice requires positive supporting evidence and
executable checks, not merely the absence of a contradiction. A weak choice
without that support remains unresolved and follows the linked-issue/blocking
rule below rather than becoming the default by inertia.
“No author code found” and “the paper omits this detail” are unresolved outcomes,
not evidence that a preferred implementation is correct. Conflicting primary
evidence cannot be silently discarded because another choice trains better.

### Verification and refutation ledger

All weaker assumptions are assigned to implementation research here; none is
deferred to an untracked future investigation. At a negative gate, stop the
dependent default or fidelity claim, preserve the result, and continue independent
work. Do not work around a refutation by silently shrinking the architecture or
switching optimizer/loss. Permitted execution optimizations such as AMP cannot
serve as evidence that a contradicted reconstruction is correct. If a dependency cannot be resolved during
implementation, retain its blocked status and create a linked issue before handoff.

| Gate | Proposed interpretation and evidence | Required verification | Negative result and required disposition |
| --- | --- | --- | --- |
| R1 Architecture | Use figure block counts `(4,12,24,36)`, growth 32, bottleneck 128, compression 0.5, standard decoder Conv3D. Strong support from Fig. 1 and §3; complete topology remains uncertain. | Inspect the figure at readable resolution; trace every channel, skip, spatial dimension, BN/ReLU and convolution. Compare text, figure, any verified author implementation, and exact parameter breakdowns. | Impossible geometry or contradictory wiring refutes that literal mapping. Revise only the contradicted choice with a written rationale. A parameter mismatch alone does not license halving depth. Do not declare the whole architecture settled just because full counts and standard decoder are implemented. |
| R2 Input construction | Contiguous 12-slice samples are a plausible interpretation of the stated `224×224×12` input and avoid the measured whole-volume depth erasure. Sample construction is omitted. | Search primary methods/supplements/verified author code for cropping, spacing and sampling. Census per-case and per-component retention, voxel/physical-volume retention, and physical coordinates across all cases. Verify full coverage independently of target labels. | Author evidence for a different construction refutes the proposed default. Any completely erased native tumor component, omitted depth slice, or violation of the predeclared boundary-loss tolerance blocks long training on that representation. Investigate in-plane tiling/resampling explicitly; do not silently increase tumor weights or use validation labels to select samples. |
| R3 Phase targets and transfer | Same three-class task in both phases is better supported by §3.2 than the repository's liver-only A. Best-A weight transfer is explicit; optimizer/scheduler reset is not. | Verify §3.1–3.2 and any author code for target remapping and transfer state. Test labels, nonzero tumor gradients, best selection and phase-boundary resume. Record the reset policy independently. | Explicit contrary author evidence changes the phase-target decision. If transfer-state evidence is absent, retain the current fresh-optimizer/reset policy only as a named provisional choice; do not claim exact reconstruction. |
| R4 Epoch and LR meaning | Literal reference: SGD LR 0.01, momentum 0.5, halve every 10 epochs, A100/B1000, ten steps per epoch. “Steps/sub-epochs” is undefined and literal decay nearly freezes training. | Inspect §3.2/4.3, loss curves and any verified training code. Translate every candidate into updates, sample coverage, LR trajectory and runtime; use the claimed 42 h only as historical context with unknown batch/workload, not a throughput target for modern hardware. | No primary resolution leaves semantics unresolved. A measured near-zero update with persistent tumor failure refutes the candidate as a useful long-run protocol, not necessarily the historical claim. Preserve the literal reference; block an expensive completion run. Do not adopt cosine, a floor, Adam or “ten full passes” as paper fact. Any such experiment needs a separate named configuration and rationale. |
| R5 Loss normalization | Eq. (2) visibly uses weighted voxel CE divided by voxel count N. Proposed implementation is unreduced weighted CE summed over valid voxels, divided by valid-voxel count; current PyTorch weighted mean divides by summed weights. | Independently transcribe Eq. (2), clarify N and label encoding, check verified author code if found. Compare loss and gradients to a manual small example, including padding. | If N or implementation evidence contradicts voxel averaging, block that default until resolved. Neither normalization should be silently compensated by an LR change. Keep normalization explicit in experiment identity. |
| R6 Metric and reconstruction | Whole-case native-grid liver/tumor Dice is the appropriate local comparison unit. Overlap blending and empty-case conventions require verification. | Check paper §4.2 and official LiTS metric conventions. Test whole-case reconstruction, liver union with tumor, strict tumor class, empty cases and voxel weighting; record local split vs challenge-test distinction. | Patch-averaged Dice, uncovered voxels, label-informed inference, or undocumented empty-case differences invalidate a comparable case-level result. Withhold incomplete-case/cohort metrics. A local holdout score never establishes the reported hidden-test score. |
| R7 Modern execution validity | Use the available RTX/Blackwell hardware and a supported modern runtime. AMP/BF16/FP16, TF32, compilation, fused kernels and activation checkpointing are permitted execution choices, not claims about the authors' implementation. | For selected optimizations, compare outputs/loss/gradients and short learning behavior against a controlled FP32 reference, with declared numerical tolerances; verify finite updates, BatchNorm/RNG behavior, memory use and stop/resume. Record hardware, precision and runtime. | Material numerical or learning regressions refute the chosen execution mode: fix or disable it before long training. OOM requires resource tuning that preserves the selected architecture and documented recipe. Missing GTX 1080 hardware or Pascal support is not a blocker, and no 8 GB cap is required. |
| R8 Augmentation and sampling details | Clipping and random scale/mirror are stated; horizontal-only flip, probability 0.5, uniform in-plane scaling, center crop/zero pad, transform order, stride 12/tail overlap and uniform probability blending are repository or proposed choices. | Seek primary augmentation/sample-construction code; document axis/frame, distribution, interpolation, padding value/validity, order and seed behavior. Test paired geometry, stochastic lesion retention and deterministic inference independently. | Contrary source evidence refutes the matching default. No evidence leaves each choice provisional or unresolved. Do not call retained repository defaults verified; label-informed inference, mismatched pairs, systematic removal of small components or unaccounted padded targets blocks the candidate. |
| R9 Framework numerics | The authors used TensorFlow/Keras; framework defaults are not a specification. Current PyTorch interpolation, BN, bias/init and optimizer details require explicit disposition. | Inventory model-internal interpolation mode/alignment, BN epsilon/momentum/update semantics, convolution bias/initialization, padding/stride rounding, SGD momentum/Nesterov/dampening/weight decay and loss reduction. Inspect source-version-specific primary docs or author code; use analytical or small cross-framework comparisons when semantics differ. | An incompatible equation/update or unsupported default refutes equivalence. Unknown author versions leave defaults provisional; choose explicit values with rationale and tests, not implicit latest-library defaults. Unresolved differences with material output/gradient effects block an exact-model claim and must be resolved or linked as issues before architecture/configuration lock. |

R1 must explicitly revisit assumptions beyond block counts: Fig. 1's dashed
skip sources appear different from the current matching-resolution skips; its
pool kernel annotation and printed output dimensions require reconciliation;
the transition ordering in the diagram differs from the current implementation
and is not fully consistent with the narrative. Preserve cropped figure references
locally and a written mapping in the decision record. Do not commit copyrighted
figure images. The current full-count profiling candidates retain current skips,
pool and transition operations and therefore do not resolve these questions.

R4 also has contradictory figure evidence: Figure 2's displayed curves both
span 0–100 and its legend names DenseUNet157 rather than DenseUNet569. Verify
these directly when recording the schedule decision. They weaken confidence in
the reporting but do not establish a replacement epoch count or decay rule.

## Historical hardware evidence and modern execution

The paper §4.3 claims a single GTX 1080 with 8 GB and about 42 h training. The old
run used batch size 6 and recorded 11,980 MiB GPU usage. That monitor figure combines
allocations/overhead and is not a peak live-tensor measurement.

This session performed six synthetic training steps at commit `230b761`, two
per candidate, on the RTX 5070 Ti. Input was `[1,1,12,224,224]`, FP32 with TF32
disabled, weighted CE and SGD momentum, under an 8 GiB PyTorch allocator cap.
Every step had finite loss and completed without OOM. No real training run or
checkpoint was updated.

| Candidate retaining current skip/pool/transition assumptions | Trainable parameters | Maximum allocated GiB | Maximum reserved GiB |
| --- | ---: | ---: | ---: |
| Current reduced counts, DS decoder | 3,523,643 | 1.518 | 1.816 |
| Figure counts, DS decoder | 10,795,323 | 1.750 | 2.043 |
| Figure counts, standard Conv3D decoder | 54,674,411 | 1.749 | 2.029 |

These measurements support investigating full-count candidates without assuming
they require more than 8 GB at batch size 1. They do not identify the authors' topology,
reproduce training accuracy, or predict GTX 1080 throughput. Largest retained
activation groups were high-resolution decoder blocks, especially up5/up4.
Changing convolution operators changes saved tensors and workspaces; parameter
count alone does not predict training memory.

The installed `torch 2.13.0+cu130` wheel lists `sm_75,sm_80,sm_86,sm_90,sm_100,sm_120`;
GTX 1080 is Pascal compute capability 6.1. CUDA 13 drops Pascal offline compilation
and library support. This explains why the measurements never established
Pascal compatibility; that compatibility is outside the implementation goal.
Use and pin a runtime supported by the available modern GPU after ordinary
dependency checks. No downgrade, Pascal-specific build or GTX 1080 test is required.

Sources: [NVIDIA legacy GPU capabilities](https://developer.nvidia.com/cuda/gpus/legacy),
[CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html),
[PyTorch packaging and legacy CUDA builds](https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337),
[allocated versus reserved memory](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management).

Validate the chosen execution configuration on the available GPU through cold
and warmed forward, backward, momentum allocation, validation and recovery/phase
transitions. Record peak allocated/reserved memory, device usage, runtime,
precision and convolution settings. Choose a resource budget and headroom for
that GPU; the historical 8 GiB profiling cap is not an implementation requirement.

Modern execution improvements do not require evidence that the authors used
them. They must preserve the selected architecture and declared training
semantics, with R7 numerical and learning checks for the chosen modes. Keep
precision/execution settings in experiment identity and checkpoint metadata.
Batch size 1 FP32 is a diagnostic reference, not a mandatory production setting.
Record microbatch, effective batch, sample exposure and update cadence when
tuning throughput. Gradient accumulation is not equivalent to a larger BatchNorm
batch; checkpoint recomputation must not silently double-update BatchNorm state.
An optimizer, loss, target, topology or scheduler change remains a training-recipe
decision subject to the research gates, not merely a Blackwell optimization.

Raw scripts, JSON, cold/warm phase measurements and logs are in the original
checkout at `models/audit-20260930/memory-profile/`. A [compact profile receipt](2026-09-30-memory-profile.json) is tracked alongside
this handoff so another checkout retains the measurements.

## Implementation slices and dependencies

### S1 Evidence and experiment contract

Own `docs/research/2026-06-21-denseunet569-architecture-decisions.md`, this
ledger, `dense_unet_3d/config.yaml`, and configuration validation in
`training/runtime.py`. First complete R1–R9 research dispositions. Keep validated
facts distinct from provisional choices; update historical decisions without
erasing why the old run differed.

Define an explicit reconstruction configuration identifying architecture,
sampling/grid, phase targets, loss reduction, schedule/step units, split manifest,
precision and microbatch. Preserve one named historical/reference configuration
for comparison. Do not introduce a large arbitrary architecture search space.
Each candidate should answer a particular unresolved evidence question.

Acceptance: every ledger row has a result or an active refutation/blocking gate;
each chosen default has a source or a clearly labeled inference. Configuration
printing describes actual sample/update/validation counts, not just epochs.
Blast radius: experiment semantics; no old-run rewrite or launch.

### S2 Model construction and modern runtime

Depends on R1/R9 dispositions; validate selected execution optimizations through R7.
Own `model/DenseUNet3d.py`, `model/building_blocks/{DenseBlock,dense_layer,
TransitionBlock,UpsamplingBlock,ds_conv}.py`, model tests, dependency/runtime
documentation, and model factory/loading portions of `cli.py`.

Make the selected architecture constructible from a small explicit configuration.
Use full figure counts and standard decoder convolutions as the initial
evidence-backed candidate, subject to R1's topology reconciliation; do not describe
the already-profiled54.7 M candidate as the final paper model. Keep the historical
reduced model identifiable for old weight evaluation rather than guessing from
parameter count. Record model configuration in best/last/recovery artifacts and
load it explicitly for eval/predict. Reject unsupported/mismatched metadata;
preserve the existing known legacy model contract without speculative migration.

Tests: exact per-stage shapes and named skip sources, dense concatenation,
convolution groups and kernels, BN/ReLU order, architecture parameter breakdown,
forward/backward finite gradients, explicit checkpoint model round trip, and
incompatible graph rejection even if tensor shapes happen to match. Remove the
3.6M tolerance band as a paper-fidelity assertion. Remeasure memory for the final
graph on the available GPU and validate the chosen execution modes through R7.

### S3 Spatial samples and streaming case reconstruction

Depends on R2/R6/R8 research; may proceed independently of the final model graph.
Own `dataset/LITSDataset.py`, `dataset/prepare_dataset.py`, spatial transforms,
a focused slab/tile indexing module if needed, `evaluation/evaluate.py`,
`evaluation/dice_score.py`, inference portions of `cli.py`, and corresponding tests.

Provisional initial representation: contiguous native-depth 12-slice windows,
shared image/mask in-plane transforms, deterministic coverage including tail
windows, and an explicit padding-validity mask. Retain native **depth** spacing
unless contrary primary evidence is found. Keep source affine/spacing, effective
model-grid spacing/FOV, and final native output geometry as distinct metadata.
Resizing a full 512×512 plane to 224×224 preserves its FOV but changes effective
in-plane sampling spacing; it cannot also preserve native voxel spacing. Store
the explicit model-index-to-source-index transform and its composition with the
source affine, including half-pixel offsets and axis order. Tests must verify
physical landmarks, not just array dimensions or a copied affine.

Test 224×224 in-plane resampling for lesion loss. The deterministic census must
identify native 3D tumor components using a recorded connectivity convention
(proposed: 26-connectivity for this engineering audit), track component identity
through all sample windows, and report original/retained component counts,
completely erased components, raw voxel counts and physical-volume retention.
Normalize comparisons for changed voxel volume; different-grid raw counts alone
are not retention ratios. Stratify results by native component size so large
lesions cannot hide small-lesion loss. **Any completely erased native component
fails R2 for the proposed representation.** Partial boundary loss must also have
an explicit acceptance tolerance justified from geometry and synthetic landmarks,
written before examining candidate census results; absent that disposition, the
full-FOV resize remains unresolved and long training stays blocked. Do not select
a tolerance after seeing which one makes the transform pass.

A tile-based fallback requires its own recorded inference, coverage test and
coordinate map. Retain the stated HU clamp; augmentation details pass R8 before
being selected. Report stochastic augmentation retention separately from the
deterministic representation census so intentional crops do not conceal fixed
preprocessing defects.

Index samples with stable `(case_id, bounds, geometry, valid_extent)` metadata;
keep loader output uniform and compatible with default collation where practical.
Start with stride 12 and append a final window starting at `D-12` when needed;
pad and mask the `D<12` case. This is a provisional coverage policy, not a paper
claim. For overlap, average class probabilities with explicit coverage weights,
restore the native H/W probability grid, then take argmax. Share this predictor
between CLI and evaluation rather than implementing two reconstruction paths.
Initial sampling should provide reproducible coverage without foreground-dependent
validation or prediction. Tumor oversampling is an optional separately identified
experiment, not a silent paper requirement. Never use target masks to crop or
select inference inputs; the existing `crop_to_liver` helper is not a solution.

Case prediction must cover every native voxel. Reconstruct one case at a time
with bounded CPU buffers, an explicit overlap aggregation rule, and argmax only
after combining predictions. Retain native affine/orientation and exclude padding.
Evaluate complete cases against their complete masks, not each slab as a “case.”
Keep partial-case/cohort results withheld when a bound is reached. Revisit CLI
evaluation limits because a patch batch count no longer equals a case count.

Tests: coordinate-coded images and class landmarks, nonmultiple depths, depth<12,
rectangular volumes, boundary tumors, all-background volumes, deterministic
coverage/order, tail/overlap normalization, native-grid round trip, perfect
predictions→Dice 1, and dataset label-retention census. No train/validation case
overlap. Preserve the current split while comparing interventions; any later
split change is a new manifest/experiment, not an unnoticed data expansion.

### S4 Phase targets, loss and update semantics

Depends on R3/R4/R5/R9 and the sample contract from S3. Own `training/loss.py`,
`training/train.py`, `training/cascaded_driver.py`, phase-loader setup in `cli.py`,
and their tests. Coordinate CLI ownership sequentially with S2/S3.

For the supported same-task interpretation, preserve labels 0/1/2 in both phases:
remove implicit folding from CLI dry/real loaders and driver `_LiverOnlyLoader`
use. Audit dataset `detect_tumors` and both managed/standalone training paths.
Make best selection consistent with the declared three-class task and retain
finite-score guards. Transfer actual selected A weights exactly once; define
optimizer/scheduler reset separately and test it at a transition boundary.

Implement the verified loss reduction explicitly, with padding excluded from
both numerator and denominator. Keep weights background 0.2/liver 1.2/tumor 2.2
unless R5 finds contrary primary evidence; do not introduce Dice/focal loss.
Manual CPU gradient tests must distinguish voxel mean from weighted mean.

Express epochs in terms of configured update/sample coverage and retain a tested
literal scheduler reference. R4 must settle or explicitly block the long-run
configuration; implementation success does not authorize thousands of stalled
updates. No unsupported schedule is to be relabeled paper-faithful.

### S5 Recovery integration and failure observability

Depends on S2–S4 interfaces. Own `training/recovery.py`, `training/runtime.py`,
runtime integration tests and `docs/training-operations.md`.

Extend `_dataset_identity`, `_loader_identity`, model/experiment fingerprints,
RNG capture and checkpoint metadata for the new sample index, padding policy,
geometry, labels/loss/schedule semantics and model graph. Preserve supported
zero-worker deterministic ordering initially; do not add custom workers/samplers
whose state cannot be restored. Hash source cases and split/sample manifests;
do not hash only repeated patches or ignore changed geometry.

Tests: uninterrupted versus resumed weights, optimizer, scheduler, RNG and sample
order agree in both phases and at transfer; incompatible old semantics fail
clearly; stop/failure/recovery/budget behavior survives slab evaluation. Reject
corrupt/partial cases without publishing partial-cohort scores.

Add per-class target/prediction counts, tumor-positive samples/cases seen, class
loss summaries and bounded parameter-update diagnostics. Report LR/gradient
scale and update size separately; changing BatchNorm statistics is not proof of
parameter learning. Flag repeated zero tumor predictions on positive targets
without declaring failure on a negative-only batch. Any diagnostic stop policy
must be explicit and recorded, not a hidden change to the published schedule.

### S6 Verification and experiment readiness

Depends on S1–S5. Run CPU tests, Ruff, formatting and mypy in the chosen supported
environment, followed by independent correctness review. Reuse passed checks
unless integration changes invalidate them. Separate GPU diagnostics from a
long-run training allocation.

With explicit GPU authorization, execute a bounded tumor-positive overfit check
on a few verified samples: record before/after loss, class2 predictions, Dice,
gradient/update norms and memory. It must produce overlapping tumor predictions
and materially improve tumor Dice, not merely decrease aggregate loss. If it
fails, stop progression to a long run and isolate data/targets/loss/graph/update
problems; do not sweep optimizers until one produces a flattering curve.

Complete R7 on the available GPU, including safe stop/resume, the final model's
memory profile, and selected precision/execution checks. Do not block completion
or create a required issue for absent GTX 1080 testing. Long training additionally requires R2 retention, R4
schedule, full-case validation and overfit gates to pass, plus a new run name,
recorded split/config/source identity and an authorized finite allocation.

## Required implementation handoff

Deliver code/PR and checks, updated evidence ledger with every negative outcome,
selected reconstruction configuration, model/parameter/shape manifest, modern
runtime and precision/execution settings, memory/overfit diagnostics, and remaining
linked issues. Report local validation separately from the paper's hidden-test
results. Do not mark the paper-reconstruction objective complete while a dependent
research or learning gate remains open. Execution checks apply to the chosen
available hardware; GTX 1080 compatibility is not an acceptance requirement.
