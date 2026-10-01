# R1/R9 topology and framework evidence

Date: 2026-09-30. Scope: architecture and model/optimizer numerical semantics.
**Exact-paper architecture remains unresolved.** The implementation supplies one
explicit diagnostic candidate, `figure_skip_reconstruction_v1`, alongside the
unchanged `historical_reduced` model. Neither parameter count nor successful
shape tests establishes reproduction. These gates block an exact-model/default
lock, not bounded CPU investigation.

## Sources and provenance

Primary paper: Alalwan et al., *Efficient 3D Deep Learning Model for Medical Image
Semantic Segmentation*, [published article](https://doi.org/10.1016/j.aej.2020.10.046),
§3 p.1233, Fig.1 p.1234, §4.3 p.1235, Table 3 p.1238. The supplied local PDF
was rendered using `pdftoppm -f 4 -l 4 -scale-to 2800 -png -singlefile` and
inspected visually. Local-only render/crop:
`/tmp/denseunet-reconstruction-research/figure1.png` and `figure1-skips.png`.
No paper images are tracked. Extracted text alone misses crucial labels.

A bounded web search used the full article title, `Alalwan DenseUNet github`,
`Nasser Alalwan github segmentation`, `Amr Abozeid DenseUNet`,
`3D-DenseUNet-569 code`, and `AlHabshy github liver segmentation`.
No verified author implementation or supplement resolving topology was found.
The search hit [this repository](https://github.com/NguyenJus/pytorch-dense-unet-3d),
a later reimplementation, which is not independent author evidence. This is an
**unresolved search outcome**, not proof that no implementation exists.

Framework evidence is versioned source, not an assertion of the authors' versions:

- [Keras 2.3.1 BatchNormalization](https://github.com/keras-team/keras/blob/2.3.1/keras/layers/normalization.py),
  especially initialization and `call`.
- [Keras 2.3.1 convolution layers](https://github.com/keras-team/keras/blob/2.3.1/keras/layers/convolutional.py),
  `_Conv`, `UpSampling3D`; [TensorFlow backend](https://github.com/keras-team/keras/blob/2.3.1/keras/backend/tensorflow_backend.py),
  `resize_volumes` and normalization functions.
- [Keras 2.3.1 initializers](https://github.com/keras-team/keras/blob/2.3.1/keras/initializers.py),
  `_compute_fans` and `VarianceScaling`.
- [Keras 2.3.1 SGD](https://github.com/keras-team/keras/blob/2.3.1/keras/optimizers.py),
  `SGD.get_updates`.
- [TensorFlow 2.3.0 tf.keras BN](https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/layers/normalization.py),
  `_moments`, non-fused `call`, and running-stat updates.
- [TensorFlow convolution/padding contract](https://www.tensorflow.org/api_docs/python/tf/nn/conv3d),
  [PyTorch SGD equations](https://docs.pytorch.org/docs/2.9/generated/torch.optim.SGD.html).

Local executable reference: PyTorch `2.13.0+cu130`, CPU only. TensorFlow is not
installed; checks below compare analytical source equations with PyTorch, **not**
a cross-framework runtime equivalence test.

## Figure transcription and disposition

Spatial triples in this record use `(D,H,W)`; the paper prints `(H,W,D)`.
Figure colors identify Conv→orange BN→green ReLU. The entire graph is not
internally consistent, so supported individual facts do not make the graph settled.

| Item | Evidence and counterevidence | Disposition and consequence |
| --- | --- | --- |
| Input/output | §3 equation (1): input 224×224×12; final feature width64; probabilities width3. | **Supported**. Candidate emits logits and leaves softmax to loss/prediction, equivalent at the probability interface. |
| Counts | Fig.1 visibly labels 4×,12×,24×,36×. | **Supported**. Half counts as a faithful mapping are **refuted**; historical compatibility alone retains them. |
| Dense connectivity | §3 explicitly describes direct dense links. Repeated growth convolution width32 in Fig.1. | **Provisionally adopted** standard running concatenation including block input: `C_out=C_in+32L`. Exact aggregate block channels are not printed reliably. |
| Bottleneck widths | DB1–3 first Conv3D label128; **DB4 first Conv3D label32**, as does its DS output. | **Supported as printed**, provenance to implementation **unresolved**. Candidate preserves `(128,128,128,32)`; uniform128 is contradicted by this label. It is not silently corrected as a typo. |
| Dense DS operator | §3 describes channelwise depthwise then 1×1 pointwise mixing; figure places BN/ReLU after the composite DS operation. | **Supported** operator/order; **provisionally adopted** depth multiplier1 (one per input), with no intervening BN/ReLU between depthwise and pointwise. |
| Stem | Figure Conv k7,s2,pad0,width96; output112×112×6; BN→ReLU. | Literal pad0 **refuted** (109×109×3). SAME_UPPER **provisionally adopted** from stated dimensions and TensorFlow provenance. Symmetric pad3 also matches shape but samples different coordinates. |
| Pool | Figure explicitly Maxpool3D k3,s2,pad0, output56×56×3. Narrative says pooling replaced by strided convolution, but figure retains this one pool. | k3/pad0 **refuted** (55×55×2). Candidate keeps figure k3 and uses SAME_UPPER. k2 historical pool is not figure-supported. Whether authors retained this pool is **unresolved** under prose/figure conflict. |
| Transitions | Figure: Conv1→BN→ReLU→Conv1_reduce(stride2), no bars after reduction. Prose lists BN,Conv1,stridedConv, and later says BN/activation after each convolution. | Figure order **provisionally adopted**, exact order **unresolved**. Historical BN→Conv→Conv is not the drawn order. Candidate compression halves aggregate channels at first Conv1, second preserves width; location of compression and printed width32 conflict remain **unresolved**. |
| Transition depth | Printed spatial dimensions preserve depth3 through all transitions while halving in-plane dimensions. | **Provisionally adopted** stride `(1,2,2)`; literal isotropic stride2 **refuted** by printed depth. |
| Dense/decoder padding | k3,pad0 paired with unchanged dimensions. | Literal valid padding **refuted**; pad1 **provisionally adopted** to preserve dimensions and concatenation. |
| Decoder widths/operators | Fig.1 layers1–5 label504,224,192,96,64 and Conv3D (not DS-Conv3D). §3 assigns DS replacement specifically to dense blocks. | Standard Conv3D with these widths **supported**. Historical DS decoder is an explicit efficiency deviation. |
| Decoder BN/ReLU | Upward arrows pass Conv then orange BN then green ReLU. §3 agrees BN/activation after convolution. | **Supported** composite Conv→BN→ReLU. |
| Classifier | Width3 and k1, but `stride=0`. | stride0 **refuted** as invalid; stride1 **provisionally adopted** from equation(1) output geometry. No classifier BN/ReLU bars; logits followed by probabilistic softmax **provisionally adopted**. |
| 569 label and counts | Paper counts heterogeneous operations, not a reproducible layer enumeration. 3.6M in prose/Table1 conflicts with 36,270,875 trainable /36,433,587 total in Table3. | **Unresolved**. No invented layer count or count fitting, tolerance band, or width/depth tuning. |

## Every dashed skip and candidate repair

The diagram's dashed paths are visibly **stem→up5, DB1→up4, DB2→up3,
DB3→up2, DB4→up1**. They land by decoder convolution after the drawn
upsampling. Dense-block source endpoints meet the right block edge around the
DS operation, rather than an unambiguous post-concatenation tensor. Thus exact
tap (last growth output, accumulated block output, or an internal pre/post-BN
feature) is **unresolved**. §3's low-level features from the opposite dense block
supports using the aggregate block output as a **provisional diagnostic choice**.

| Decoder | Main source shape/channels | Figure skip source shape/channels under concatenation | Target / output channels |
| --- | --- | --- | --- |
| up1 | DB4 `(3,7,7)` /1660 | DB4 `(3,7,7)` /1660 | `(3,14,14)` /504 |
| up2 | up1 `(3,14,14)` /504 | DB3 `(3,14,14)` /1016 | `(3,28,28)` /224 |
| up3 | up2 `(3,28,28)` /224 | DB2 `(3,28,28)` /496 | `(3,56,56)` /192 |
| up4 | up3 `(3,56,56)` /192 | DB1 `(3,56,56)` /224 | `(6,112,112)` /96 |
| up5 | up4 `(6,112,112)` /96 | stem `(6,112,112)` /96 | `(12,224,224)` /64 |

Direct concatenation of the literal skip tensors after resizing only the main
path is **refuted by geometry at every level**. Matching-resolution historical
sources avoid that conflict but contradict all five dashed sources (including
omitting up5's drawn stem skip). The candidate repairs geometry by independently
resizing both branches to the stated output before concatenation. Positive
support: the figure identifies these source/target blocks, while text requires
interpolation followed by feature combination. Missing skip-resize operators
and the duplicate DB4 inputs at up1 are explicit counterevidence/ambiguities.
This is a **provisionally adopted diagnostic repair**, not a discovered author
operation. Concat-before-resize is another algebraically equivalent expression
when branches share size and interpolation; it does not resolve source taps.

Encoder channels are 96→224→112→496→248→1016→508→1660. Candidate transitions
apply compression0.5 to aggregate channels, contradicting literal width32 in
the diagram. Treating every transition as width32 instead would contradict
textual compression0.5 under dense concatenation. No hidden projection is added
to force both. This conflict remains open.

## R9 explicit numerical inventory

| Choice | Historical implementation | Candidate / disposition |
| --- | --- | --- |
| Interpolation | trilinear, align_corners=True, main only | trilinear, align_corners=False, both paths. **Provisionally adopted** continuous 3D extension of stated bilinear operation and half-pixel geometry; exact coordinate rule **unresolved**. “Bilinear” cannot directly consume a 5D PyTorch volume. Keras2.3.1 `UpSampling3D` repeats elements (nearest), so treating a Keras default as evidence of stated bilinear is **refuted**. |
| BN epsilon | implicit1e-5 | explicit1e-3, positively supported by both inspected Keras-era APIs, **provisionally adopted**. Author overrides/version **unresolved**. |
| BN momentum | PyTorch new-stat weight0.1 | explicit0.01, corresponding to Keras old-stat weight0.99, **provisionally adopted**. Same literal momentum numbers across APIs would be wrong. |
| BN running variance | PyTorch unbiased estimator for running state; biased batch variance during training | Explicitly retain PyTorch unbiased running convention. **Unresolved author equivalence**, diagnostic convention only. Standalone Keras2.3.1 corrects population variance with n/(n−1−eps), tf.keras2.3 nonfused uses population variance. Merely matching epsilon/momentum does not settle this. |
| BN affine/state | gamma1,beta0,mean0,var1, tracked statistics | Same explicit convention; supported by inspected APIs but author settings **unresolved**. |
| Bias | stem/bottleneck/transition False, DS True, classifier True | all Conv biases present and zero initialized; **provisionally adopted** from versioned Keras Conv defaults. Paper mentions biases but does not identify individual layers. |
| Initialization | PyTorch Conv reset: Kaiming uniform a=√5 and sampled bias | Glorot uniform standard convs; zero biases. Depthwise uses Keras-style `(kD,kH,kW,input_channels,multiplier)` fan calculation, multiplier1. **Provisionally adopted framework-based convention**; unknown custom 3D DS implementation means author equivalence **unresolved**, especially depthwise initialization. |
| Padding/rounding | symmetric stem3; pool2 valid; transition stride122 | SAME_UPPER stem k7 and pool k3; stride1 k3 symmetric1; transition stride122. Shapes agree, coordinates differ. **Provisionally adopted** framework-informed repair, exact author padding **unresolved**. Compression floors but all selected widths are even so floor/round coincide. |
| SGD | PyTorch SGD, momentum0.5; implicit dampening0,NesterovFalse,weight_decay0 | Paper states LR/momentum values; the SGD algorithm itself is provisional (see recipe evidence). NesterovFalse/no weight decay/dampening0 are **provisional**, not known author settings. Velocity equation equivalence across LR changes is **refuted**; see below. Model files do not change optimizer. |
| Loss | PyTorch class-weighted mean divides by sum of target weights | Eq.(2) uses voxel-count denominator; R5 owns implementation. The two reductions differ in gradient scale; no automatic LR compensation. |

For input length12,k7,stride2, TensorFlow SAME_UPPER uses `(left,right)=(2,3)`;
symmetric PyTorch pad3 yields the same output6 but shifts centers by one voxel.
For even length,k3,stride2 SAME_UPPER uses `(0,1)`; historical k2 pools a different
neighborhood. For transition k1,stride2, output is ceil(in/2), not an arbitrary
floor-half convention. Current selected in-plane sizes are even.

Keras2.3.1 SGD: `v_t=mu*v_(t-1)-lr_t*g_t`, `p_t=p_(t-1)+v_t`.
PyTorch (dampening0): `b_t=mu*b_(t-1)+g_t`, `p_t=p_(t-1)-lr_t*b_t`.
With p0=1, two gradients1, mu=.5 and LRs .1,.05, Keras equation yields .8,
PyTorch .825. At constant LR they agree under corresponding initial states;
at decay boundaries they generally do not. Do not assert equivalent SGD merely
because LR and momentum numbers match. Author TensorFlow/Keras version and
optimizer implementation require an issue before exact recipe lock.

## Executable checks and architecture artifacts

Implementation is in `model/reconstruction.py`; strict named identities,
factory, manifest, fingerprint and metadata validation are in `model/config.py`.
Historical `DenseUNet3d()` graph/state keys are unchanged. Metadata validates
canonical complete configuration, including operations that keep tensor shapes
unchanged; it does not guess identity from weight counts. Missing legacy metadata
must take a caller's explicitly known historical path.

[Model manifest](2026-09-30-model-manifest.json) records exact stage shapes,
parameters, skips and complete numeric/config identities. Totals:

- Historical: **3,523,643** trainable parameters.
- Figure-skip candidate: **64,591,723** trainable parameters.

Candidate largest stage is up1,45,180,072 parameters, because concatenating two
1660-channel tensors feeds standard3×3×3→504. This follows the named graph and
is not evidence for or against the paper's conflicting counts.

CPU tests in `tests/model/test_reconstruction.py` check full-resolution meta
stage shapes/operators, historical tensor checkpoint round trip, candidate meta
checkpoint round trip, altered-graph/fingerprint rejection, explicit SAME
sampling coordinates, dense concatenation, small-tensor finite forward/backward
including skip gradients, BN equation/state, and LR-boundary SGD discrepancy.
Meta checks prove constructibility/shape, **not** numerical behavior or memory.
A separate full fixed-input CPU run at `(1,1,12,224,224)` completed candidate
forward/backward in 7.76 seconds with finite output, input gradients and all
parameter gradients. This check is also retained as a test. Finite synthetic
gradients do not establish learning or R7.
No GPU operation or training was launched.

Verification receipt: `pytest -q tests/model/test_reconstruction.py
 tests/model/test_dense_unet_3d.py` completed successfully: **16 passed**, including
both historical and candidate full-input finite forward/backward tests. Full
output is `/tmp/denseunet-reconstruction-research/model-all-tests.log`.
Ruff passed on all four changed/new model/test Python files; `git diff --check`
passed on owned paths. Mypy passed for both new model modules with
`--python-version 3.13`; the repository default3.11 invocation is blocked by
installed NumPy stub syntax, not a source check pass. No code changed after
these checks except this verification receipt.


## Blocking issues required before exact-model claim

1. R1: aggregate channels vs printed32, DB4 bottleneck32 vs usual128, dashed
   source taps and missing resize, transition ordering/compression placement,
   pool narrative/figure disagreement. Need verified implementation or author
   clarification; retain candidate status until resolved.
2. R9: author framework versions/custom 3D operators, interpolation coordinates,
   padding origin, BN variance convention, bias/init, and optimizer velocity at
   scheduled LR changes. Analytical counterexamples refute blanket equivalence.
3. R7 remains separate: no full candidate GPU memory/precision/learning/resume
   certification follows from CPU graph and unit checks.

The primary integration handoff must link active repository issues for unresolved
R1/R9 dependencies; this record alone is not an architecture-lock waiver.

Open topology/framework gate: [issue #16](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/16).
