# Phase, loss and update evidence — 2026-09-30

This record resolves reconstruction gates R3/R5 at the level of the published
mathematics and records an unresolved R4 long-run gate. It does not establish the
authors' actual implementation. The stopped overnight run is preserved. No GPU
job or training restart was executed for this research.

## Independently inspected primary evidence

The local PDF `docs/papers/alalwan2020.pdf` in the original checkout was rendered
with `pdftoppm` and visually inspected at published pp. 1234–1235. The rendered
pages are temporary files under `/tmp/denseunet-reconstruction-research/`; no
copyrighted figure is included in the repository. Text extraction was checked
against these rendered pages, rather than used as the equation authority.

Eq. (2), p. 1234, transcribes to

\[
L(y,\hat y)=-\frac1N\sum_{i=1}^N\sum_{c=1}^3
w_i^c y_i^c\log\hat y_i^c.
\]

Its leading sign is minus; its denominator is **N**, not the sum of weights.
The accompanying text identifies i as a voxel, y as ground-truth probability,
y-hat as predicted class probability, and w as a class weighting factor. It does
not separately define N in prose or state batch-versus-volume averaging.
Eq. (1), p. 1233, describes a three-channel output and a class-index target per
voxel. Thus one-hot ground-truth probabilities and N equal to the number of
contributing voxels is the mutually consistent interpretation. Label indices
0/1/2 are a repository encoding of the paper's background/liver/tumor classes.
The adjacent sentence gives weights background 0.2, liver 1.2, lesion 2.2.

Section 3.2, pp. 1234–1235, describes initial training of the same model for 100
epochs, 10 steps (sub-epochs) per epoch; optimal initial weights become the base
for a final phase of 1000 epochs, 10 steps per epoch. It does not describe changing
the label task, a separate liver network, optimizer-state transfer, optimizer or
scheduler reset, checkpoint-selection metric, batch size, validation split, or
sample count per step. Section 4.3 states momentum 0.5, initial LR 0.01 and a
factor-of-two decrease every ten epochs. SGD is a reasonable interpretation of
this momentum description, but the optimizer algorithm itself is not explicitly
named in that paragraph. TensorFlow/Keras is named; versions are absent.

Figure 2, p. 1235, independently reads as follows: horizontal axis **Epoch**,
ticks 0/20/40/60/80/100; vertical axis **Loss**, approximately 0 through 1. Blue
legend names `3D-DenseUNet-157 1st training phase`; red legend names
`3D-DenseUNet-157 2nd training phase`. Both curves extend to epoch 100. The
caption names 3D-DenseUnet-569. The blue curve starts near 0.97, red near 0.57;
both settle near 0.05–0.08. These approximate visual values are not digitized
data. The mismatched model label and final-phase axis conflict with §3.2;
neither establishes a replacement epoch count or LR rule.

Primary paper: [Alalwan et al., DOI 10.1016/j.aej.2020.10.046](https://doi.org/10.1016/j.aej.2020.10.046),
[official article record](https://www.sciencedirect.com/science/article/pii/S1110016820305639).

## Author implementation search and limits

Searches on September 30 used exact title, DOI, `3D-DenseUNet-569`, all author
name/code combinations most relevant to the surfaced candidate, and GitHub
repository search for `DenseUNet`, `DenseUNet-569`, and `alalwan`. No verified
author implementation of this paper was located. This is an unresolved search
outcome, not evidence that the repository's defaults match.

The exact-name web search surfaced the current
[NguyenJus reconstruction](https://github.com/NguyenJus/pytorch-dense-unet-3d);
its authorship and README identify it as a later reconstruction, so it is not
independent primary evidence. The related
[H-DenseUNet repository](https://github.com/xmengli/H-DenseUNet) belongs to another
paper and cannot settle Alalwan's recipe. A possible Amr Abozeid account,
[amrapozaid](https://github.com/amrapozaid), lists only two public fire/smoke
repositories, confirmed through authenticated `gh api users/amrapozaid/repos`.
A matching name is insufficient to authenticate it as this paper's author,
and neither repository is the liver model. No communication was sent to authors.
The official article web fetch returned HTTP 403; the complete locally available
PDF was inspected and contains no visible source-code URL or supplementary
training implementation. Unindexed/private code or a separately hosted supplement
could still exist.

## Disposition ledger

| Choice | Disposition and positive evidence | Counterevidence, uncertainty and dependent gate |
| --- | --- | --- |
| Three-class targets in both phases | **Supported reconstruction**: §3 defines the three-class model, §3.1 the three-class objective, and §3.2 continues that model with learned weights. | An unstated historical remapping cannot be ruled out without author code. Exact historical target implementation remains unverified. Controls S4 target preservation. |
| Liver-only A as paper fact | **Refuted as a source-supported claim**: no such phase-task change is described; it changes the published joint objective and suppresses class 2. | Its possible contribution to the old collapse is plausible, not isolated by the audit. Historical mode remains available only under its own identity. |
| Best/optimal A model-weight transfer | **Supported** by §3.2. | Definition of optimal and whether BN state is included are omitted. Exact selected full state-dict transfer is the local reproducible implementation; balanced foreground validation Dice is **provisionally adopted**, consistent with the joint task but not stated by the authors. |
| Fresh optimizer and LR scheduler in B | **Provisionally adopted** as named `best_weights_fresh_optimizer` policy: §3.2 explicitly calls for weight transfer, not optimizer transfer; existing implementation is tested. | Absence of a transfer statement is not proof of reset. R3 exact-state fidelity remains unresolved. |
| SGD LR 0.01/momentum 0.5 | Values **supported**, SGD algorithm **provisionally adopted** from momentum context and established repository choice. | Algorithm/version, Nesterov, dampening and decay are absent. Explicit PyTorch options are local inference. |
| Ten minibatches per epoch | **Unresolved** historical meaning; named literal reference is executable. | Modern Keras calls a training batch a step, but “sub-epochs” and Table 2's epoch/step workload wording weaken a unique reading. No author code found. R4 blocks long training. |
| A100/B1000 and factor 0.5 every ten epochs | **Supported as a literal text reference**, **unresolved as actual historical protocol**. | Figure 2 conflicts; run audit refutes the current candidate as a useful continuation protocol, not the text's historical truth. Do not substitute B100, cosine, a floor or Adam as paper facts. |
| Voxel-count weighted CE | **Supported mathematical reconstruction** by Eq. (2). | N's batching scope is omitted. Equal-size unpadded inputs make pooled voxel averaging and per-volume averaging equivalent. Variable valid counts require a declared local choice. Controls S4 loss. |
| PyTorch weighted mean equals Eq. (2) | **Refuted** analytically and by CPU loss/gradient comparison. | PyTorch class-index CE mean divides by sum of valid target weights, not valid voxel count. No LR compensation is justified. |
| Padding excluded from loss/count | **Provisionally adopted local extension** to preserve Eq. (2)'s observed voxel objective. | Paper does not discuss padding. Explicit mask/ignore index and tests are required; artificial padding cannot silently become background supervision. |
| PyTorch momentum equals unknown Keras momentum | **Unresolved exact equivalence**, named `pytorch_gradient_buffer` remains local inference. | LR-scaled Keras velocity and PyTorch gradient buffer differ at LR changes; see optimizer numerical evidence in the topology research record. Do not guess an author version. |

## Loss and gradient executable check

CPU float64 experiment used four zero-logit voxels with targets `[0,0,1,2]` and
weights `[0.2,1.2,2.2]`. The weighted numerator is `3.8*log(3)`.
Voxel mean gives `1.0436816742347044`; weighted mean gives
`1.0986122886681098`. Every nonzero voxel-mean gradient equals 0.95 times its
weighted-mean counterpart (`sum weights/N = 3.8/4`). The tumor-logit gradient at
the tumor voxel is `-0.3666666666666667`, so gradient descent raises its logit.
With three additional ignored/padded voxels, those same four-voxel values must
remain unchanged and padding gradients must be zero. An all-invalid batch has
no mathematical mean; reject it instead of producing a zero or NaN update.

[PyTorch CrossEntropyLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
describes the class-index weighted mean; the installed implementation was also
inspected through its executable reduction. This normalization factor depends
on class prevalence, so a constant LR correction does not generally reproduce
voxel mean.

## Candidate update, exposure, LR and runtime accounting

Let S be the number of training samples (cases under whole-volume construction,
slabs under slab construction), b the microbatch, L=ceil(S/b) loader batches with
`drop_last=False`, and t the measured update seconds on the selected hardware.
No sample is a uniquely exposed case unless construction makes that true.

| Explicit candidate | A/B updates | Exposure and validation | Update-only runtime |
| --- | --- | --- | --- |
| Ten minibatches per logical epoch | 1,000 / 10,000 | At most 1,000b / 10,000b sample presentations; short final batches reduce this. Coverage is presentations/S, not guaranteed unique coverage. 1,100 scheduled epoch validations at cadence 1. | 11,000t |
| Ten complete loader passes per logical epoch | 1,000L / 10,000L | 1,000S / 10,000S presentations; 1,000 / 10,000 complete sample passes. | 11,000Lt |
| Figure-inspired B100 × ten minibatches | 1,000 / 1,000 | At most 2,000b presentations, 200 epoch validations. **Unsupported experiment**, not an adopted correction. | 2,000t |

For the audited S=28 whole cases, b=6, L=5, ten minibatches cycle the loader twice
per epoch: 56 case presentations, not 60, because each pass ends with a four-case
batch. Across A/B this is 5,600/56,000 presentations (200/2,000 complete passes).
Ten full passes instead would mean 50 updates and 280 presentations per logical
epoch: 55,000 total updates and 308,000 presentations (11,000 complete passes).
With a future slab construction S changes substantially; recompute counts from
its actual manifest before launch. The unmanaged `train()` API currently uses
one complete loader pass per epoch, while `_run_epoch` uses a fixed minibatch
count; their epoch units must be printed explicitly.

Under fresh per-phase reset and end-of-epoch StepLR, update LR for phase epoch e
is `0.01 * 0.5**floor((e-1)/10)`. A100 last updates use `1.953125e-5`, leaving
scheduler LR `9.765625e-6`. B173 uses `7.62939453125e-8`; B1000 last updates use
`1.5777218104420236e-32`, leaving `7.888609052210118e-33`. Halving occurs every
100 updates under ten minibatches, versus every 100L updates under ten passes.
The endpoint LR remains negligible under either reading. Carrying the A decay
counter into B would further reduce its endpoint; this is another unverified
policy, not the retained reference.

The paper's 42-hour statement implies 13.75 seconds per update only if the
11,000-minibatch reading holds **and all runtime is attributed to updates**.
Validation, compilation and I/O invalidate that simple identity. Table 2 reports
10.4 seconds per epoch/step on average (5/10/15/20-case inputs, workload/batch
undefined); `11,000*10.4` is 31.78 hours before overhead. This arithmetic neither
proves nor refutes a step interpretation. Full-pass interpretation would scale
update-only time by L; unknown batch/sample workload prevents a historical
throughput comparison. Modern measured runtime is independent of that GPU claim.

## Code mapping and blockers

At the audited starting state, `training/loss.py` returns weighted-mean
`CrossEntropyLoss`; `training/train.py` specifies PyTorch SGD/StepLR and traverses
one full loader each unmanaged epoch. `training/cascaded_driver.py` uses ten
minibatches, reopens its loader at epoch start, wraps A training/validation in
`_LiverOnlyLoader`, selects A by liver Dice and B by mean foreground Dice, then
loads A best state dict into a fresh optimizer/scheduler. The CLI independently
constructs A datasets with `detect_tumors=False`; changing only the driver would
not repair managed CLI targets. Managed recovery owns a separate phase transition
and selection path and must adopt the same explicit contract. Historical defaults
must not change old experiment identity by accident.

R4 remains a material long-run blocker: the audited B172–173 parameter delta had
relative L2 norm `1.06e-9`, only 1.86% of elements changed, and tumor Dice stayed
zero. Geometry/labels/loss fixes justify a new bounded diagnostic, not thousands
of literal-decay completion epochs. Before any long run, settle or link an issue
for step/schedule interpretation and demonstrate tumor-positive overfit with
finite class-2 gradients, predictions and meaningful updates. R3 reset and R9
optimizer version uncertainty additionally block an exact reconstruction claim.

Related local evidence: [training run audit](2026-09-30-training-run-audit.md),
[implementation handoff R3/R4/R5 and S4](2026-09-30-paper-reconstruction-handoff.md).
Primary framework context: [Keras training API](https://keras.io/api/models/model_training_apis/)
and [SGD update rule](https://keras.io/api/optimizers/sgd/). Current framework docs
explain possible semantics; they are not proof of the authors' historical calls.

## Implemented S4 checks

The explicit choices are implemented as opt-in `phase_a_targets=three_class`,
`loss_reduction=valid_voxel_mean`, `phase_transfer_policy=best_weights_fresh_optimizer`
and `optimizer_semantics=pytorch_gradient_buffer`; omission retains historical
liver-only/weighted-mean behavior. Native slabs require the three-class target
contract. Both reductions exclude `-100`; dictionary `valid_mask` maps padding
to that ignore index. An all-invalid batch and invalid valid labels are rejected.
Reconstruction experiments are rejected by unmanaged public training APIs so
that they cannot bypass managed budgets and evidence gates.

The focused CPU loss/train/cascade suite passed **72 tests** (37 expected warnings
about unmanaged historical APIs) in 2.74 seconds. It includes independent
analytical gradients for both denominators and padded voxels, nonzero tumor
gradients, three-class A label/selection checks, exact selected A transfer into a
fresh optimizer/scheduler, a managed real-NIfTI native-slab batch with padding,
and nonfinite-loss/gradient rejection before an update. A three-step analytical
SGD test agrees with LR-scaled velocity at constant LR and differs at the first
halving: PyTorch parameter 0.6625 versus Keras-style velocity parameter 0.625,
starting at 1 with constant gradient 1, LR 0.1/0.1/0.05 and momentum 0.5.
This verifies the distinction, not an author's version choice.

Optional per-epoch diagnostics report valid target/prediction class counts,
weighted CE sums per class, tumor-positive sample exposure, global gradient norm,
and an update norm for at most 4096 sampled parameter entries by default.
The update norm is explicitly sampled, not a whole-model delta or proof of
learning. Checkpoints created by these training APIs carry model/experiment
metadata when available. These CPU checks do not establish a usable learning
schedule, lesion retention across the cohort, or tumor overfit performance.

Open schedule/transfer/optimizer gate: [issue #17](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/17).
