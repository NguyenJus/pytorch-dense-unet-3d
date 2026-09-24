# Repository and paper audit — 2026-09-23

## Source and scope

This audit compares the model with Alalwan et al., “Efficient 3D Deep Learning
Model for Medical Image Semantic Segmentation,” *Alexandria Engineering
Journal* 60 (2021), 1231–1239, [doi:10.1016/j.aej.2020.10.046](https://doi.org/10.1016/j.aej.2020.10.046).
Page and figure references below refer to the published paper; no paper content
is copied into this repository.

## Resolved documentation and model issues

| Finding | Evidence | Resolution |
| --- | --- | --- |
| The repository called decoder DS-Conv paper-faithful. | The paper states that standard convolution **in each dense block** is replaced with DS-Conv (p. 1233); Fig. 1 labels decoder operations `Conv3D`. | Kept DS-Conv decoder blocks because the existing reduced architecture chose them for its parameter budget, but label them an implementation deviation in code and README. |
| The README instructed users to set unsupported `dataset_path` and implied an automatic holdout. | Current loader configuration requires `pathing.train_img_dirs` and separate `pathing.test_img_dirs`. | Updated README data setup. |
| Invalid model shapes failed later during the fixed-size decoder path. | The model contains fixed targets `(12, 224, 224)` and the paper's input example is `224 × 224 × 12` (p. 1233, Fig. 1). | Added a clear NCDHW input-contract error and CPU test. |
| Several claims presented a modeled full-depth parameter total as proof of a paper contradiction. | The paper reports 3.6 M parameters (p. 1233) and Fig. 1 shows block counts `(4, 12, 24, 36)`, but omits enough implementation details to reproduce its exact count. | README now distinguishes paper facts from this repository's reconstruction assumptions. |

## Independent parameter check

Using the current PyTorch module definitions, real dense concatenation, block
counts `(4, 12, 24, 36)`, compression `0.5`, and this repository's DS-Conv
decoder mapping yields **10,795,323** trainable parameters. The shipped
`(2, 6, 12, 18)` mapping yields **3,523,643**. These are reproducible counts
for this implementation, not measurements of the authors’ TensorFlow/Keras
model and not proof that the paper is internally inconsistent.

`tests/model/test_dense_unet_3d.py::test_full_depth_reconstruction_parameter_count`
constructs the full-depth module without a forward pass and pins the
full-depth count. The per-module table in the architecture decision record is for the
shipped reduced model.

## Remaining limitations

- The paper does not fully specify channel propagation after dense
  concatenation, padding/bias behavior, or decoder concatenation widths.
  Exact architectural reproduction and attribution of the 3.6 M count remain
  unresolved.
- The paper describes bilinear interpolation in its upsampling block (p. 1233)
  while this 3D PyTorch implementation uses trilinear interpolation. This is a
  necessary framework-level 3D interpretation and is documented as such, not a
  literal operator match.
- The shipped model is intentionally reduced depth and therefore must not be
  reported as a reproduced 569-layer model or as reproducing the paper’s LiTS
  results.
- The paper reports two training phases of 100 and 1000 epochs, with 10 steps
  per epoch (pp. 1234–1235), but does not define “step.” The repository uses a
  configurable number of loader iterations per epoch; this is a documented
  interpretation, not a verified reconstruction.
- Fig. 1 prints padding values that do not produce its displayed spatial sizes
  under standard convolution arithmetic (for example, the stride-2, kernel-7
  stem shown from 224 to 112). The implementation selects padding and 3D
  interpolation targets that preserve the figure's displayed dimensions. This
  is a geometric reconciliation, not an author-verified setting.

## Engineering corrections

- Correct the NIfTI-to-CDHW axis conversion through the complete dataset
  preprocessing path; test voxel positions, not only tensor shapes.
- Pair image/mask files by case ID, including compressed NIfTI files, and reject
  incomplete pairs, grid mismatches, and overlapping train/validation cases.
- Preserve discrete mask labels during scale augmentation and preserve
  rectangular image shapes.
- Honor inference intensity configuration and preserve NIfTI spatial metadata.
- Preserve depth-one class-index targets; reject empty training loaders and
  nonpositive cascaded schedules instead of hanging or producing no checkpoint.
- Select best checkpoints only on finite scores with strict improvement;
  explicitly fail when validation never produces a usable score, preventing
  reuse of a stale best checkpoint. Bare checkpoint filenames now work.
- Enforce mypy's exit status in CI, include the example config in built wheels,
  and expose the cascaded schedule in the example config.

The preprocessing correction changes the inputs seen by existing weights.
Historical checkpoints and images are not validation of the corrected pipeline;
retraining and real LiTS validation remain necessary. No GPU training or real
LiTS accuracy measurement was performed during this audit. Code review is
intentionally deferred at the owner's request.

## Validation

CPU-only verification on Python 3.11:

- Full suite: 211 tests passed; two subsequently added inference-input
  preprocessing tests also passed (213 tests covered in total).
- Ruff lint and formatting checks: passed after formatting corrections.
- Mypy: passed across the package.
- Source distribution and wheel builds: passed; wheel includes `config.yaml`.
- Standalone wheel import, configuration-resource lookup, and CLI help: passed.
- Git whitespace check: passed.

Tests use synthetic data and include actual full-model CPU forward passes;
they do not establish liver or lesion segmentation accuracy.
