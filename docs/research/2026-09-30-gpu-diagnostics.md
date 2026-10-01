# Bounded GPU diagnostics

**failed diagnostic gates**. 90/90 updates; 301.525/900 seconds cumulative GPU allocation.

Figure-skip reconstruction candidate (64,591,723 parameters), RTX 5070 Ti, FP32, TF32 disabled, cuDNN deterministic=True and benchmark=False; global strict determinism=False. This is diagnostic evidence, not exact paper reconstruction.

Warm full-shape batch `(1,1,12,224,224)`:

| Stage | Seconds | Peak allocated GiB | Peak reserved GiB | Device used GiB |
| --- | ---: | ---: | ---: | ---: |
| forward | 0.147 | 2.397 | 2.820 | 4.134 |
| backward | 0.442 | 2.115 | 2.820 | 4.134 |
| momentum_update | 0.018 | 0.739 | 2.820 | 4.134 |

Device usage is sampled after each stage; allocator peaks are measured within each stage.

- Preserved attempt 1: nll_loss2d_forward_out_cuda_template does not have a deterministic implementation
- Preserved attempt 2: max_pool3d_with_indices_backward_cuda does not have a deterministic implementation
- Preserved attempt 3: Stop/resume or repeated-step numerical tolerance failed

Managed phase-boundary stop/resume: **fail**. Floating outputs/losses/all gradients/model/BN/optimizer/history use predeclared `atol=1e-6, rtol=1e-4`; RNG, scheduler and integer counters require exact equality. Four uninterrupted updates were compared with two updates, a committed phase-B boundary stop, then two resumed updates. No validation data or metrics were used for parity.

Tumor-positive overfit: **evaluation readiness inconclusive; acceptance criterion unmet**, 80 updates, fixed SGD/CE recipe. First two train slabs exceeding 1% tumor were case `-16`, native depth `[396,408)` and `[408,420)`, with 12,435 and 17,343 tumor voxels (2.065% and 2.880% of 602,112 valid voxels each). Eval loss 6.043518 → 0.321154; mean tumor Dice 0.021452 → 0.000000. Final class2 prediction counts were [740, 766], with tumor-overlap counts [0, 0]. The predeclared gate requires overlap on both and at least 0.10 absolute Dice gain.

Train-forward mean Dice over the first/last two steps was 0.057002 → 0.869205. These predictions demonstrate training-mode memorization. Evaluation readiness was not demonstrated within 80 updates; the small near-zero Dice decline alone does not establish degradation or model failure. The predeclared acceptance flag remains false. The cause of the train/eval discrepancy remains unresolved; BN statistics are a candidate to investigate, not an established diagnosis.

All 64,591,723 parameter elements were included in gradient/update norms: gradient L2 [0.667454, 35.298870], actual update L2 [0.007947, 0.352989]. All parameters/gradients/updates remained finite. All 161 BN counters advanced once per training step, and evaluation preserved every BN buffer.

Held-out validation eval smoke used one target-independent first slab of case `-100`; logits/loss were finite and BN buffers preserved. This negative slab is not a tumor-learning check or full-case validation result.

Limitation: final overfit weights were not persisted by the exercised runner. A comparison of batch-stat and running-stat forwards on identical final weights was not executed, so BN causality remains untested. No updates were repeated. The runner now saves final weights for a future authorized allocation.

Long-run gate: **blocked by R7 repeated-step/resume numerical divergence; diagnostic overfit cannot clear execution gate**. The independent R2/full-case/research gates remain open.

The adjacent JSON contains full counts, all update norms, cold/warm memory/runtime, source/config/split/script hashes and numerical replay deviations. Exact source for both update-bearing attempts and their config is tracked in the [artifact archive](artifacts/README.md); zero-update and reporting-only source is represented by hashes only. Raw cached slabs, complete per-step outputs/gradients, recovery checkpoints and failed-attempt reports/logs remain local-only under `models/reconstruction-20260930/gpu/`; they are not available from a clean clone.
