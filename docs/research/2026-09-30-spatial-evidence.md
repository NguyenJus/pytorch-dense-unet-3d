# Spatial reconstruction evidence and gates (2026-09-30)

This records R2/R6/R8 before any candidate dataset census. No retention results
were inspected when declaring the geometric tolerance below. Implementation and
census are engineering tests, not evidence of the authors' undocumented recipe.

## Primary sources and author-code search

- **P**: Alalwan et al., published pp. 1231–1239,
  [DOI](https://doi.org/10.1016/j.aej.2020.10.046); local untracked PDF
  `/home/justin/projects/pytorch-dense-unet-3d/docs/papers/alalwan2020.pdf`.
  §3.1 p.1233 specifies 224×224×12 training samples and three-class output;
  §4.1 p.1235 specifies HU clipping and random scale/mirror;
  §4.2 p.1235 gives case and global Dice. No construction, orientation,
  distribution, padding, sliding-window, interpolation, or blending algorithm
  is specified in those sections or elsewhere in the inspected paper.
- **L**: Bilic et al., [LiTS benchmark, 2019 v1 §3.5.1 pp.14–15](https://arxiv.org/pdf/1901.04056v1),
  cited as P reference 23. Global Dice pools masks across cases into one volume;
  case Dice averages across all cases. Later arXiv versions revise reporting,
  so use the historically cited version for P's global-score interpretation.
- **N**: [Official challenge evaluation notebook](https://github.com/PatrickChrist/LITS-CHALLENGE/blob/c0648928131970cc2a053d0be8c1cf4d6b7e934e/evaluation_notebook.ipynb),
  `get_scores` and complete-volume loop. It uses liver `>=1`, lesion `==2`,
  and calls MedPy `dc`; it has no exclusion of both-empty cases. L identifies
  this repository as the official metric implementation.
- **M**: [Official requirements](https://github.com/PatrickChrist/LITS-CHALLENGE/blob/c0648928131970cc2a053d0be8c1cf4d6b7e934e/requirements.txt)
  pin MedPy 0.2.2.
  [MedPy 0.2.2 source](https://github.com/loli/medpy/blob/ff272f521fc47688b6e27ae96f7fb3f6f7fcede8/medpy/metric/binary.py#L30)
  returns 0 for a zero Dice denominator. Current MedPy differs; importing the
  latest package would not reproduce this historical empty convention.

On 2026-09-30 searched exact paper title, DOI/model name plus code/GitHub,
Nasser Alalwan plus GitHub/DenseUNet, Amr Abozeid plus GitHub/segmentation,
and AbdAllah ElHabshy/AlHabshy plus GitHub. No implementation with a verified
author-to-paper provenance chain was located. The search found this project's
NguyenJus reimplementation, which is not author code. An `amrapozaid` repository
claims an Amr Abozeid smoke/fire project, but neither identity continuity to this
paper nor DenseUNet code was established; it is not evidence for this model.
Publisher/DOI web fetches failed; the complete local published PDF was inspected,
but a separately hosted supplement could not be ruled out. **Author code and
supplement availability remain unresolved**, not absent by proof. Resolution
requires an author/institution/publisher link to the exact artifact, followed by
comparison against P. No author contact was sent.

## Choice dispositions

Each provisional item has positive engineering evidence and a falsifiable check.
This support permits implementation for investigation, not an exact-paper claim.

| Choice | Disposition, positive evidence and counterevidence | Required executable check / downstream gate |
| --- | --- | --- |
| Input 224×224×12, output labels 0/1/2 | **Supported**, P §3.1; axis order is not specified. | Shape/class contract; record NIfTI source axes separately from tensor CDHW. |
| Entire case resized to depth 12 | **Refuted as acceptable representation** by the prior run audit's native lesion erasure; P does not specify this construction. | Preserve historical implementation only for comparison; never infer complete lesion coverage from shape. |
| Contiguous native-depth windows, no depth resampling | **Provisionally adopted engineering candidate**: preserves every slice and physical depth coordinates by identity mapping, while matching P's depth extent. Paper construction itself **unresolved**. | Every native slice covered for D<12, D=12 and nonmultiples; affine landmark agreement and padding exclusion. |
| Stride 12 plus final start D−12 | **Provisionally adopted**: interval arithmetic proves full coverage and bounds overlap to the tail. No primary claim of this stride. | Exhaustive small-depth coverage/order checks; stable manifest; no duplicates except intentional overlap. |
| Full-FOV H/W resize to 224 | **Unresolved for selection**: matches stated input size and preserves FOV mathematically, but paper omits resize/crop/spacing and nearest sampling can erase small lesions. | Census and predeclared tolerance below; any erased component blocks selection and long training. |
| Native in-plane tiles | **Unresolved alternative**, no P evidence. Can become provisional engineering fallback only if resize fails and complete tile geometry/coverage is tested. | No label-derived bounds; complete H/W coverage, padding validity, native coordinate reconstruction; own census. |
| Source affine/spacing retained; effective model spacing H/224 and W/224 times native spacing | **Supported geometry**, from an explicit affine composition, not a paper preprocessing claim. Keeping original affine alone for a resized tensor is **refuted**. | Physical coordinate-coded landmarks, rectangular input and oblique affine; half-pixel translation and axis permutation. |
| Trilinear image / nearest-exact labels, shared half-pixel map | **Provisionally adopted**: categorical nearest retains class IDs, common coordinates prevent image/mask offset; corrected repo baseline supplies operational support. Author kernels **unresolved**. | Synthetic ramps/labels and component identity round trip; exact source-index enumeration. |
| Depth padding masked, target −100 | **Provisionally adopted**: exclusion keeps synthetic voxels out of loss and metrics, while satisfying fixed tensor shape. No P padding specification. | Padded voxels have zero loss gradient; one-slice source returns one native slice. |
| HU clipping [−200,250] | **Supported**, P §4.1. | Values below/above bounds clamp; labels unaffected. |
| Random scale limits 0.8–1.2 and mirroring | **Supported**, P §4.1, with awkward combined wording. Mirror itself has no numerical scale. | Shared geometry and controlled RNG; validation deterministic. |
| Uniform factor, in-plane-only scale, one common H/W factor | **Unresolved as author choices**. Bounds alone give no distribution/axis evidence. Repo comment claiming paper-matching in-plane augmentation is unsupported. | Keep disabled in the investigated slab candidate until separate explicitly named augmentation contract is selected; seeded pairing and stochastic retention audit then required. |
| Horizontal tensor-W-only flip, probability 0.5 | **Unresolved**: mirror supports reflection generally but does not establish this axis/probability or anatomical frame. | Axis-to-source/world map and seeded distribution; no silent default adoption. |
| Center crop/zero pad after scale; clamp→resize→flip→scale order | **Unresolved**: no source for center, pad intensity, order or padding semantics. Zoom-in removes edge objects. | Separate stochastic per-component survival audit; validity transformed jointly; excluded padding; boundary landmark cases. |
| One RNG draw per sample, shared image/mask transform | **Provisionally adopted engineering requirement**: random augmentation and paired supervised labels require stochastic decisions and matching geometry. Author RNG/seed unspecified. | Repeat-seed exact replay, varying decisions, paired coordinate landmarks and recovery RNG checks. |
| All-case native-grid Dice, liver union tumor, strict tumor | **Supported**, P §4.2, L and N. Patch-as-case scores **refuted** as comparable case scores. | Complete case before scoring; perfect nonempty masks→1, liver/tumor fold tests; interruption withholds cohort score. |
| Case Dice includes both-empty with 0 | **Supported historical N/M**; P does not state empty policy. Existing presence-aware exclusion **refuted for historical notebook equivalence**. | Truth table both empty→0, one empty→0, equal nonempty→1; declare metric version. |
| Global Dice raw voxel pooling; globally empty→0 | Pooling **supported**, L; zero result **provisionally adopted** by extending M's same denominator rule. Official notebook does not implement cohort aggregation. | Unequal case sizes expose mean-vs-pooling; no physical-volume weighting unless separately named metric. |
| Uniform probability averaging over overlap, inverse resize then argmax | **Provisionally adopted engineering candidate**: averaging normalized probabilities preserves simplex and equal tail coverage; hard votes discard uncertainty. Author aggregation **unresolved**. | Exact coverage weights, conflicting overlap example, simplex sum, no argmax before overlap/inverse resize, uncovered voxel rejection. |
| Ground-truth-informed crop or inference sample selection | **Refuted** for deployable evaluation: uses unavailable test labels; handoff explicitly forbids it. | Predictor signature accepts image geometry only; permuting target cannot change predictions/coverage. |
| Local holdout compared as hidden-test reproduction | **Refuted**: P/L use 70 hidden test cases, this checkout has a distinct labeled split. | Retain exact split manifest and report local validation identity. |
| 26-connectivity for native tumor audit | **Provisionally adopted engineering convention**: includes face/edge/corner native adjacency and preserves one stable native ID per connected set; not an official lesion metric claim. | Synthetic diagonal contacts vs separated components; never re-identify only after resize. |

## Predeclared partial-boundary tolerance

This is a geometric envelope, **not a percentage chosen from observed lesions**.
It permits only discretization near the in-plane boundary. Erasure is forbidden
even when a component is entirely boundary. It cannot establish clinical adequacy
or paper fidelity; report size-stratified actual losses alongside pass/fail.

For an axis with native extent n and model extent m, downsample selects
`s(j)=min(n−1,floor((j+0.5)n/m))` (nearest-exact). Native-grid categorical
round trip selects `t(i)=s(min(m−1,floor((i+0.5)m/n)))`.
Declare `r=max_i |t(i)−i|` by exhaustive integer enumeration of the geometry,
before loading any labels. It is bounded by `ceil(n/(2m)+0.5)` native voxels;
use the exact enumerated r. Depth radius is zero. For a component C, define its
core as per-slice binary erosion with a `(2r_H+1)×(2r_W+1)` rectangle, treating
outside the source FOV as background. Its inner boundary shell is C minus core.
Acceptance requires **zero missing core voxels**, **all lost native round-trip
voxels inside this shell**, and **at least one surviving model voxel for every
native component**. Thus component lost physical volume is bounded by the shell's
native voxel count times `abs(det(source_affine[:3,:3]))`; no cohort average may
hide a violation. Native round-trip retained voxels are identity-specific, not
the union of nearby tumors. Report model physical volume independently as
`model_component_voxels * native_voxel_volume * (H/224)*(W/224)`; window overlap
must never double-count it. Report signed volume change, native overlap retention,
shell volume and completely erased components separately.

The bound follows directly: a native core voxel's selected source index is at
most r_H/r_W away on the same depth slice and therefore lies in the same C.
This makes shell loss a conservative resampling allowance and core loss evidence
of implementation/geometry error. Synthetic axis ramps, rectangles, FOV-edge
objects, one-voxel/diagonal components and oblique/anisotropic physical landmarks
must verify the bound before the first real census. A passed envelope with large
thin-object loss does not justify relaxing zero erasure or claim useful learning.

## Required outcomes before a long run

Run deterministic census on every existing train/validation case without changing
the split; log connectivity, transforms, source geometry, stable component IDs,
all slice coverage, counts, native/model physical volumes and size strata. Keep
augmentation off for this census. Any component erasure, missing slice, missing
core or unmasked target padding blocks R2. Unresolved augmentation specifics
block an exact augmentation claim; investigate them in a separate named contract.
Full-case reconstruction tests and historical Dice truth-table tests gate R6.
Record negative gates rather than replacing loss/class weights to compensate.

Downloaded source copies and paper text are under
`/tmp/denseunet-reconstruction-research/` (untracked); no copyrighted figure or
paper binary is committed. Implementation/census receipts are appended only
after the predeclared rules above.

## Implementation receipt (before real census)

The opt-in `dataset.sampling: native_slabs` path implements the declared
full-FOV resize candidate; legacy dataset and legacy presence-aware evaluation
remain available under their original behavior. Slab mode rejects liver-only
targets and unresolved augmentation flags, uses padding target −100 and a
validity mask, and returns collatable image/target/geometry dictionaries.
`NativeSlabDataset.manifest()` records unique source cases, grids and the stable
sample index; preparation checks the existing experiment split manifest.

`evaluation.predict.predict_volume` accepts only a source image and spatial
configuration, returns native HWD uint8 labels, and keeps at most one depth
window of CPU probability planes. Overlap is averaged before bilinear native
in-plane probability restoration and argmax. Native evaluation bypasses sample
batch metrics and scores complete source cases with the declared historical
empty policy. Its `max_batches` argument counts native cases and withholds all
cohort metrics if exceeded; late corrupt targets and interruptions raise.

`scripts/census_reconstruction.py` audits each native 26-connected component,
retains its ID through exact nearest grids, excludes duplicated slab overlap,
and reports actual physical-volume ratios, native overlap retention, erasure,
boundary/core losses and size strata. Exit 2 indicates a completed negative R2
gate. No real census was run by the spatial worker before declaring tolerance.

Checks: the 12 new CPU geometry/retention/reconstruction tests passed before
releasing the census command. The complete dataset/evaluation suite then passed
92 tests. Focused Ruff checks/formatting passed. Focused mypy passed for the five
changed source modules with `--python-version 3.13` (the installed NumPy stubs
use Python 3.12 type syntax; the repository's configured 3.11 target cannot
parse those installed stubs). A legacy target-fold fixture now disables random
scale/flip so stochastic singleton erasure cannot make a label-contract test
flaky. GPU checks were not run.

## Native in-plane tiling feasibility after the R2 negative gate

The parent's ongoing real census reported six completely erased native tumor
components in validation case `-101`. One such component is sufficient to
**refute full-FOV 224×224 resize as the selected long-run representation** under
the predeclared R2 gate. The partial-boundary tolerance remains unchanged.
The following investigation does not change the selected configuration, sampling
code, or paper-fidelity disposition.

**Native in-plane 224×224 tiling is provisionally supported as an engineering
fallback for further implementation**, with positive evidence from explicit
full-coverage interval geometry and executable exact reassembly tests. Its use
by the authors remains **unresolved**: P does not specify tiling or patch FOV.
This choice preserves native sampling and avoids categorical downsampling, but
changes the anatomical context presented to the model. Native H/W tile extent
is 224 times the respective source spacing rather than the whole 512-plane
FOV; native depth extent remains 12 source slices. This is not evidence that
training the smaller FOV produces good liver/tumor predictions.

The investigated policy takes starts `0, window, 2*window, …` while the whole
window fits, appends `extent−window` if necessary, and uses start zero with
explicit valid extents for dimensions shorter than a window. For each 512-axis,
starts are `[0,224,288]`. Their intervals cover every native pixel; the final
two windows intentionally overlap by 160 pixels. Cartesian H/W products make
nine tiles per depth slab. Stable iteration order is depth start, H start,
W start and uses no target information. An index `(d,h,w)` maps to original
HWD coordinates `(h+start_H,w+start_W,d+start_D)`. The homogeneous
model-to-source matrix is

```text
[[0,1,0,start_H],
 [0,0,1,start_W],
 [1,0,0,start_D],
 [0,0,0,1]]
```

Compose this with the source affine for model-to-world coordinates. There is
no in-plane scale or half-pixel resize translation; affine column lengths and
voxel determinant retain native spacing and voxel volume. Depth, H and W
padding must all receive validity false and target −100; CT padding intensity
must be declared independently. Padding is excluded when scattering output.
At inference, sum class probabilities and valid coverage weights at identical
native indices, divide once every contributing tile/slab is present, and then
argmax. No spatial interpolation is needed. A streaming implementation can
complete all H/W tiles for one depth start before flushing slices preceding the
next start, retaining at most 12 full native probability planes plus weights.

The header-only audit read all cases from the fixed split manifest, validated
image/mask grids without accessing image or target voxel arrays, and counted:

| Fixed split | Cases | Full-FOV resize slabs | Native tiles | Sample-count ratio |
| --- | ---: | ---: | ---: | ---: |
| Train | 28 | 1,257 | 11,313 | 9× |
| Validation | 98 | 3,588 | 32,292 | 9× |
| Total | 126 | 4,845 | 43,605 | 9× |

All 126 source planes are 512×512; train depth ranges 75–861, validation depth
74–987. The ninefold count estimates model forwards/one-complete-coverage sample
exposure at unchanged batch size, **not measured runtime**. If steps stay fixed,
the fraction of source case coverage per epoch changes substantially; schedule
and update semantics must be re-audited rather than silently keeping a claimed
equivalent epoch. CPU buffering grows from model-resolution to native-resolution
probability planes; inference/training memory and throughput need measurement.

`scripts/audit_native_tiling.py` executed six synthetic cases: singleton,
sub-window rectangular, exact window, nonmultiple dimensions, 512×512×25, and
301×117×17. Unique voxel IDs reassembled exactly after weighted overlap, with
no uncovered coordinate and no padding leakage; each synthetic 26-connected
tumor ID/count remained unchanged. Maximum geometric coverage was eight when
depth and both in-plane tails overlapped. An oblique anisotropic affine landmark
passed the explicit index permutation/translation check. Source-ID identity
implies any categorical tumor component is retained under exact oracle-label
tiling/reassembly: each native voxel occurs in at least one valid tile and
returns to that identical native coordinate. It does not imply a trained model
will predict that voxel or resolve a component cut by a tile boundary.

Checks: executable synthetic audit passed; Ruff and focused mypy with Python
3.13 passed. The full header receipt was local-only at
`/tmp/denseunet-reconstruction-research/native-tiling-feasibility.json` and is
not available from a clean clone.
Reproduce with:

```sh
OMP_NUM_THREADS=2 PYTHONPATH=. /home/justin/projects/pytorch-dense-unet-3d/.venv/bin/python scripts/audit_native_tiling.py --config configs/reconstruction-reference.yaml --output /tmp/native-tiling-feasibility.json
```

**Production selection stays blocked** at the linked [R2/S3 gate](2026-09-30-paper-reconstruction-handoff.md#s3-spatial-samples-and-streaming-case-reconstruction).
The following was the feasibility-stage follow-up list; its integration and
deterministic real-label census items are completed by the implementation
receipt below. Learning/FOV and production-selection gates remain open.

Follow-up must implement native tile indexing/validity/coordinate metadata,
source-manifest identity and streaming probability reconstruction; independently
test boundary tumors, small dimensions and interrupted/corrupt cases; execute a
full real-label identity census; and validate model learning/FOV effects with an
authorized overfit diagnostic. Retention proof permits that investigation, not
automatic default adoption. The parent's issue ledger must link this remaining
production and learning gate before final handoff. No GPU job was run.


Final real census: 28 train / 98 validation cases; 4/138 training and 18/731
validation components erased, with no omitted depth slices or boundary-core
failures. Full-FOV resize is refuted for long training. See the
[tracked summary](2026-09-30-retention-census-summary.json) and
[required spatial follow-up #18](https://github.com/NguyenJus/pytorch-dense-unet-3d/issues/18).
The header audit found 106 sources declaring mm and 20 with unknown units; absolute
mm3 for unknown-unit sources is an explicit provisional LiTS inference, while
per-case retention ratios and erasure are unit-independent. Future census runs
record that assumption and convert declared meter/micron units explicitly.

## Native tile engineering implementation (#18)

`dataset.sampling: native_slabs` now accepts the opt-in named
`dataset.inplane_representation: native_tiles_v1` with `resize_img: false`.
Omitting the representation retains the existing `full_fov_resize` contract;
reference/default configurations are unchanged. The isolated
`configs/reconstruction-native-tiles-diagnostic.yaml` retains the recorded
28/98 split, three-class targets, loss reduction and class weights, and sets
CPU execution. This configuration is an engineering diagnostic, not approval
for learning or a long run. No GPU or learning job was launched for this work.

Indexing follows the predeclared depth/H/W Cartesian order with the final
native tail starts and includes every source coordinate. `NativeSlabDataset`
returns CDHW tensors with DHW start/valid extents, model-to-source/world maps,
and the same collatable training dictionary/manifest contracts. NIfTI array
axes remain explicit; anatomical orientation is not inferred. The native map
contains permutation and integer translation only, composed with the complete
source affine, including obliquity/shear. Native model voxel volume equals the
source determinant. Image padding is constant `clamp_hu_range.min` (−200 by
default), including when clipping is disabled; target padding is −100 and
validity is false for all three padded axes. Padding contributes no loss or
reconstruction weights.

The shared predictor reads only source image geometry and bounded image tiles.
It completes all H/W tiles for one depth start before flushing slices that no
future depth start can cover. Its active buffer contains at most one depth
window of three-class full native probability planes and integer per-pixel
coverage weights, plus the current tile; the final native uint8 label volume
is retained. Every valid tile probability is scattered onto the identical
source coordinate, divided by its actual positive coverage, then argmaxed.
No native interpolation or target-selected bounds are used. Missing coverage,
nonfinite image/logits, incompatible logits, or interruption withhold a complete
prediction; complete-case evaluation forwards the representation and preserves
historical empty-case Dice semantics.

Native preprocessing checkpoint identity records the complete named spatial
geometry and `coordinate_grid: native_tiles_v1`; inference rejects a resize or
tile-size mismatch. Recovery manifests already fingerprint the full geometry,
source files and ordered samples, so a native/resize or index change cannot
silently resume. Legacy preprocessing identity remains unchanged.

The augmentation decision is explicitly `disabled_unresolved`. Mirror axes,
mirror probability, scale distribution, anatomical frame, padding after scale,
and stochastic transform order have no selected contract. With both flags false,
there are no random geometric transforms or stochastic retention claims. The
implemented deterministic order is native image extraction, optional HU clamp,
then high-end D/H/W padding; categorical extraction/padding shares those bounds.
Enabled scale or mirror flags are rejected. Paper fidelity and any later
augmentation implementation/retention audit remain separate open gates.

The native real-label census uses production `tile_bounds`, `tile_slices`, and
`categorical_tile` extraction/padding for stable native 26-component identities.
It verifies all voxel coverage, exact identity reassembly, padded-target validity,
and unique native component volumes, excluding overlap duplication. The same
predeclared zero-erasure rule remains; native geometric radius is zero and
there is no permitted native voxel loss. `--aggregate-only` emits split-level
counts, physical volumes and size strata without source paths, affine grids,
per-case component arrays or patient imagery. Full logs remain local/untracked.

Reproduce the CPU census on the recorded locally available dataset:

```sh
OMP_NUM_THREADS=2 PYTHONPATH=. python scripts/census_reconstruction.py \
  --config configs/reconstruction-native-tiles-diagnostic.yaml \
  --aggregate-only --output /tmp/native-tile-census18.json
```

Independent tests use explicit tail start expectations, categorical/unique
coordinate reassembly, native 26-connected components, rectangular/sub-window
padding, oblique affine landmarks, probability-conflict overlap, missing coverage,
interruption, complete-case metrics, checkpoint mismatches, and zero padded loss
gradients. Learning/FOV effects, context across tile boundaries, throughput/memory
and schedule exposure remain unmeasured. Passing deterministic retention opens
investigation of those gates and does not close #18 or select production defaults.

### Completed native real-label census

The existing CPU census completed successfully with **exit status 0** and
`gate_pass: true`. The tracked
[aggregate receipt](2026-09-30-native-tile-census-summary.json) contains no
source paths, per-case grids, component arrays or patient imagery.

| Fixed split | Cases | Native 26-components | Erased components | Core/boundary failures | Covered native slices | Native tiles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 28 | 138 | 0 | 0 | 14,918 / 14,918 | 11,313 |
| Validation | 98 | 731 | 0 | 0 | 42,518 / 42,518 | 32,292 |
| Total | 126 | 869 | 0 | 0 | 57,436 / 57,436 | 43,605 |

Every production-extracted native component identity reassembled exactly, with
positive coverage for every source voxel and no invalid padding contribution.
All size strata retained every component and exactly 100% of native component
volume. Unique native tumor counts were 2,484,416 train and 15,791,080 validation
voxels; overlap was excluded from volume counts. Native/model/retained physical
volumes agree. As recorded for this identical split, 106 source headers declare
mm and 20 omit spatial units: their absolute mm3 values use provisional LiTS
millimeter provenance, while exact retention and coverage are unit-independent.

Executed command (Python 3.11 CPU dependency environment):

```sh
OMP_NUM_THREADS=2 PYTHONPATH=. /tmp/dense-unet-pr-ci311/bin/python \
  scripts/census_reconstruction.py \
  --config configs/reconstruction-native-tiles-diagnostic.yaml \
  --aggregate-only --output /tmp/native-tile-census18.json
```

The session returned exit 0; its local log is `/tmp/native-tile-census18.log`.
The census command emits the measured aggregate fields, not the complete tracked
receipt. After successful completion, `exit_code`, `status`,
`physical_units_disposition` and `command` were added as archival metadata; the
measured fields match the local raw output. The later `provenance` metadata pins
the recoverable original PR #22 head, package tree, census script, config and
ordered split manifest. The physical-unit note is curated from the pinned
[prior retention summary](2026-09-30-retention-census-summary.json): its 126
per-case `source_spatial_units` entries reproduce 106 mm / 20 unknown and its
`verified_source_spatial_units` totals; the native aggregate command does not
emit those counts. That revision captures source **after the run**; it is
not a recorded execution revision. Dataset image/label hashes and the transient
CPU environment are not archived in this aggregate receipt, so these bindings
identify the available source/config/split without proving identical raw inputs
or dependencies in a future rerun.

To recover the pinned implementation, use commit
`25d6b6ff16723b69ce9b56207855888876d66454` in a separate checkout and verify the
SHA256 values in `provenance.files` against its file bytes before rerunning the
command above with a suitable CPU interpreter and the same local dataset.
For archival augmentation, return to the final PR checkout containing the new
`provenance` metadata and use the original preserved raw output at
`/tmp/native-tile-census18.json`. The pinned original commit does not contain
that metadata. Reconstruct the historical receipt there with:

```sh
python - <<'PY_ARCHIVE'
import json
from pathlib import Path

receipt = json.loads(Path(
    "docs/research/2026-09-30-native-tile-census-summary.json"
).read_text())
raw = json.loads(Path("/tmp/native-tile-census18.json").read_text())
# Reuse the recorded metadata only to reconstruct this historical receipt.
# A new run needs its own observed exit code, command and provenance.
for key in ("exit_code", "status", "physical_units_disposition", "command", "provenance"):
    raw[key] = receipt[key]
Path("/tmp/native-tile-census18-archived.json").write_text(
    json.dumps(raw, indent=2) + "\n"
)
PY_ARCHIVE
```

This post-processing preserves measured fields and reconstructs the tracked
receipt from the original raw output; it is not part of the census executable.
The full real-label deterministic retention result supports this named native
representation's zero-erasure geometry requirement, replacing the refuted resize
candidate for further investigation. It does **not** establish useful predictions,
learning/FOV adequacy, paper augmentation fidelity, or measured runtime/memory.
Those gates remain open; #18 is not closed and production defaults remain gated.
