# Architecture Decision Record — 3D-DenseUNet-569

**Date:** 2026-06-21
**Spec:**
`docs/superpowers/specs/2026-06-21-faithful-3d-denseunet569-design.md`
§3, §6 G1, §8
**Code:**
`dense_unet_3d/model/DenseUNet3d.py`,
`dense_unet_3d/model/building_blocks/TransitionBlock.py`

---

## 1. Fig 1 values overridden and why

### 1.1 Per-block channel labels ignored

Fig 1 prints "32" as the channel count at almost every block boundary.
That contradicts dense connectivity: each dense layer concatenates its
32-channel output onto the running feature stack, so the channel width
after block *k* is `initial_channels + k × growth_rate`.
With growth rate 32 and the implemented block counts (see §4) the
running-concatenation math governs; the literal "32" labels are ignored.

**Resolution (spec §3 rule 2):** Fig 1's per-block channel labels are
explicitly non-authoritative. The running-concatenation math governs;
the literal "32" labels are ignored.

### 1.2 Padding values recalculated

Fig 1 annotates `pad=0` on several convolutions where that is
geometrically impossible:

- **Stem Conv3D k=7 s=2:** `pad=0` would map 224 → 109, not 112.
  The correct value is `pad=3`, which yields
  `floor((224 + 2×3 − 7)/2 + 1) = 112`.
  Same for the depth axis: `floor((12 + 2×3 − 7)/2 + 1) = 6`. ✓
- **Dense-layer and decoder k=3 same-resolution convolutions:**
  `pad=0` would shrink spatial extent by 2 per side per conv.
  Same-resolution requires `pad=1`.

**Resolution (spec §3 rule 3):** Padding values are chosen to produce
the spatial dimensions stated in the §2 table, overriding the figure
annotations. All intermediate spatial dims are asserted by
`tests/model/test_dense_unet_3d.py::test_intermediate_spatial_dims_match_paper`.

---

## 2. Compression target actually used

**Paper:** compression factor 0.5 (halve channels at each transition).
**Implemented:** `compression = 0.5` (exact paper value, no deviation).

`TransitionBlock` computes `out_channels = round(in_channels * 0.5)`.
With the implemented half-scale block counts (see §4), the transition
channel widths are:

| Transition | Input channels | Output channels |
| ---------- | -------------- | --------------- |
| T1 (after DB1) | 160 | 80 |
| T2 (after DB2) | 272 | 136 |
| T3 (after DB3) | 520 | 260 |

No deviation from the paper's stated 0.5 was introduced at the
`TransitionBlock` level.

---

## 3. Skip-connection wiring (5-level pairing)

The decoder starts at DB4 (bottom, 7×7×3) and ascends through five
trilinear-upsample blocks.
Each block upsamples the main path only to the target size; the skip
tensor is already at the target resolution and is concatenated unchanged
(the old both-paths-upsampled bug is fixed in `UpsamplingBlock`).

| Decoder block | Target (D×H×W) | Skip source | Skip channels |
| ------------- | -------------- | ----------- | ------------- |
| up1 | 3×14×14 | DB3 output | 520 |
| up2 | 3×28×28 | DB2 output | 272 |
| up3 | 3×56×56 | DB1 output | 160 |
| up4 | 6×112×112 | stem output | 96 |
| up5 | 12×224×224 | *none* | 0 |

**up5 has no encoder skip.** There is no encoder feature at 224×224×12
— the stem already downsampled to 112×112×6 before any dense block.
The implementation passes an empty zero tensor of shape
`(N, 0, 12, 224, 224)` so the `UpsamplingBlock` code path is uniform.

Depth restoration happens in up4 (3→6) and up5 (6→12) via the
trilinear `size=` argument, matching the paper's upsampling spatial
progression.

---

## 4. Block-count scaling decision

**Paper:** block counts (4, 12, 24, 36) at 1:3:6:9 ratio.
**Implemented:** (2, 6, 12, 18) — half-scale, preserving the 1:3:6:9 ratio.

The paper specifies block counts (4, 12, 24, 36) as an authoritative
architecture fact.  Full-scale with DS-Conv decoder and compression 0.5
yields approximately **10.4M** trainable parameters — approximately 3×
the ~3.6M ±15% target band (3.06M–4.14M).  The excess is structural:
the dense-connectivity bottleneck (in_ch → 128 per layer, where in_ch
grows by 32 each layer) means each successive DenseLayer adds a
linearly-growing first-conv cost.  DB3 alone accounts for ~2.1M params
and DB4 for ~5.2M at full scale.

Per spec §3 (amended to authorize block-count scaling as a secondary
reconciliation knob when compression + DS-Conv + skip-projection knobs
are insufficient), the block counts were halved to **(2, 6, 12, 18)**,
which:

1. Preserves the paper's 1:3:6:9 architectural ratio.
2. Reduces the bottleneck concatenation depth proportionally.
3. Achieves **3,523,643** trainable parameters — within the
   [3.06M, 4.14M] target band.

The resulting encoder channel widths (DB outputs and transition outputs)
under half-scale counts:

| Stage | Output channels |
| ----- | --------------- |
| DB1 (2 layers) | 160 (= 96 + 2×32) |
| T1 | 80 |
| DB2 (6 layers) | 272 (= 80 + 6×32) |
| T2 | 136 |
| DB3 (12 layers) | 520 (= 136 + 12×32) |
| T3 | 260 |
| DB4 (18 layers) | 836 (= 260 + 18×32) |

The code constant is `_BLOCK_COUNTS: tuple[int, int, int, int] = (2, 6, 12, 18)`
in `dense_unet_3d/model/DenseUNet3d.py` (see module-level comment).

---

## 5. Decoder convolution: DS-Conv

Each decoder upsampling block (`UpsamplingBlock`) uses a 3D
depthwise-separable convolution (DS-Conv: depthwise 3×3×3 + pointwise
1×1×1) instead of a dense 3×3×3 Conv3d, consistent with the paper's
DS-Conv-throughout design (§1 headline contribution).

This substantially reduces the decoder parameter budget: for example
`up1` with combined input 836 + 520 = 1356 channels and output 504
channels costs ~1.3M with DS-Conv vs ~19M with a dense 3×3×3 Conv3d.

---

## 6. Actual achieved parameter count

**Test command:**

```text
pytest -q tests/model/test_dense_unet_3d.py::test_param_count_within_band
```

**Printed output:**

```text
[DenseUNet3d] trainable parameters = 3,523,643
```

**Status: PASSING.**
The test asserts `3,060,000 ≤ params ≤ 4,140,000` (3.6M ±15%).
The implementation produces **3,523,643** trainable parameters — within
the [3.06M, 4.14M] band.

Per-submodule breakdown:

| Submodule | Parameters |
| --------- | ---------- |
| stem | 33,120 |
| dense\_block1 | 44,736 |
| transition1 | 19,520 |
| dense\_block2 | 171,072 |
| transition2 | 56,032 |
| dense\_block3 | 575,616 |
| transition3 | 203,840 |
| dense\_block4 | 1,370,304 |
| up1 | 722,904 |
| up2 | 196,224 |
| up3 | 85,056 |
| up4 | 36,000 |
| up5 | 9,024 |
| classifier | 195 |
| **Total** | **3,523,643** |

---

## 7. "10 steps per epoch" interpretation

The paper states training runs for "100 epochs (each epoch = 10
steps)" and "1000 epochs (each 10 steps)". The term "step" is not
precisely defined.

**Interpretation adopted (spec §2, training scheme):**
"10 steps per epoch" is treated as a configurable sub-epoch loop count
— the trainer iterates over the data loader 10 times per epoch call.
This is the literal reading of the paper's language.
The implementation exposes this as a config key (`steps_per_epoch`,
default 10).

The cascaded 2-phase structure:

- **Phase A:** 100 epochs × 10 steps; best weights saved by val Dice.
- **Phase B:** reload Phase A best; 1000 epochs × 10 steps.

The spec notes this is "underspecified in the paper" and explicitly
documents it as an interpretation, not a precise reverse-engineering of
the original training loop.
See `dense_unet_3d/training/cascaded_driver.py` for the
implementation.

---

## 8. Summary of Fig 1 deviations

| Fig 1 element | Printed value | Override | Reason |
| ------------- | ------------- | -------- | ------ |
| Per-block channel labels | "32" everywhere | Concat math | Dense conn. |
| Stem padding | `pad=0` | `pad=3` | k=7 s=2 requires pad=3 for 224→112 |
| Dense-layer k=3 padding | `pad=0` | `pad=1` | Same-resolution requires pad=1 |
| Compression factor | 0.5 | 0.5 (unchanged) | No deviation needed |
| Block counts | (4,12,24,36) | (2,6,12,18) half-scale | Full-scale yields ~10.4M params; half-scale achieves 3.52M within ±15% band |
