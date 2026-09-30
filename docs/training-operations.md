# Recoverable training operations

Training is a finite two-phase schedule. An invocation is a separate, explicitly
bounded allocation. Nothing in this workflow automatically restarts a process or
launches training after a reboot. A real GPU smoke and any subsequent GPU run
require separate owner authorization; issue #11 does not authorize either.
Keep the stopped run and its existing checkpoints unchanged. Choose a new run
name for a new experiment.

## Launch and budget

Set dataset directories, output paths, and a unique `pathing.run_name` in a copied
configuration. The run directory is `<model_save_dir>/<run_name>`.

```bash
dense-unet-3d preflight --config config.yaml --full-decode
# Example: an 8-hour attempt within a 60-hour cumulative allocation.
# Execute only after authorization for the configured device.
mkdir -p logs
nohup dense-unet-3d train --config config.yaml \
  --wall-seconds 28800 --budget-seconds 216000 --max-retries 2 \
  >> logs/run-console.log 2>&1 &
```

`--dry-run` uses small synthetic data and a tiny model on CPU regardless of GPU
configuration. It still creates a real durable run directory; use an isolated
configuration/run name rather than the stopped experiment's path.

The CLI prints resolved epochs, updates, validation cadence/pass counts and stop
conditions before preflight/model/device allocation. It prints the effective
persisted budget after acquiring ownership and discovers validation case/batch
counts before decoding the dataset. Preflight, source/data identity checks,
allocation, training, validation, checkpoint I/O and requested final evaluation
count. A standalone `preflight` command is outside that training allocation.

Both limits use elapsed wall time, not CUDA-active time. Invocation limits may
change on resume; the cumulative limit is persisted and cannot change/reset on
resume. Omitting `--budget-seconds` on resume retains the original limit. `null`
configuration limits mean unbounded, so supply finite limits for overnight work.
A clean stopped attempt charges its actual elapsed duration. An attempt lost
without terminal evidence is conservatively charged through explicit recovery,
including downtime; clock rollback refuses recovery. Budget exhaustion never
implicitly creates a fresh experiment or grants another allocation.

Stop requests take effect at a completed epoch boundary. Training/validation
already in progress finishes before the recovery snapshot commits. Worst-case
latency is the remaining epoch updates plus scheduled full validation and
checkpoint I/O; a blocked kernel or filesystem can make that latency unbounded.
Preflight is also cooperative at its completion, rather than interrupted midway.
These limits are safe-stop requests, not hard process-kill deadlines. Allow this
margin in scheduler allocations. SIGKILL cannot commit a new snapshot.

## Monitor and stop

```bash
dense-unet-3d status --run-dir models/example_run
# A separate cheap process can observe for an hour, without a chat session.
dense-unet-3d status --run-dir models/example_run --watch \
  --interval 10 --max-seconds 3600 >> logs/status.jsonl
# Verify the local process identity and request SIGTERM through a pidfd.
dense-unet-3d stop --run-dir models/example_run
```

`runtime.json` contains stable run identity, unique attempt identity, resolved
configuration, source identity, persistent allocations/attempt history, ownership,
latest progress and heartbeat. `events.jsonl` appends machine-readable events;
console redirection above also appends. Epoch events report phase, epoch, global
update count, metrics, training/validation seconds and phase ETA even on a
validation plateau. ETA includes sample count and observed timing range; the
other phase remains explicitly unmeasured until timings exist. The observed range
is not a confidence interval or a guarantee.

Status reports `heartbeat_age_seconds`, `progress_age_seconds`, verified ownership
and stall indication. Configure `runtime.stall_seconds` according to measured
worst-case epoch/validation time. A live heartbeat with old progress can indicate
a blocked batch, validation or storage operation. Monitoring reports stalls and
does not relaunch training. A missing/reused PID or lost ownership produces
`unknown` when no terminal outcome was committed; it never invents success.
Stop refuses unknown/stale/foreign ownership, preventing signals to unrelated PIDs.
Local Linux `/proc`, pidfds and a filesystem with working `flock`, atomic rename,
file/directory `fsync` are required. Cross-host shared filesystem coordination is
not supported.
If Python omits its pidfd bindings, verified stop uses the corresponding libc
functions. If neither interface is available, stop refuses to signal a numeric PID.

`completed`, `user stopped`, `budget exhausted`, `failed` and `unknown` are
separate durable terminal outcomes. Clean completion and requested stops exit
normally; exceptions exit nonzero and preserve the exception type/message in a
failure event. Check the terminal reason rather than treating exit zero as full
schedule completion. OOM, nonfinite training loss/gradients/logits, validation
exceptions, low free storage and checkpoint/heartbeat write failures stop with
failure evidence when storage permits it. `runtime.min_free_bytes` reserves a
configured free-space floor before checkpoints. Storage exhaustion can prevent
logging the final failure itself; retain stderr and interpret missing terminal
evidence as unknown. Undefined presence-aware class Dice is represented as null
in JSON, and is valid when that class has no evidence; Phase A liver-only labels
can have undefined tumor metrics without optimization failure.

## Resume and crash/reboot recovery

```bash
# After an ordinary safe stop, use the same experiment config/run path.
dense-unet-3d resume --config config.yaml --wall-seconds 28800 \
  >> logs/run-console.log 2>&1
# After failure, SIGKILL, host reboot or lost terminal evidence:
dense-unet-3d status --run-dir models/example_run
# Inspect events and console logs, fix the external cause, then explicitly recover.
dense-unet-3d resume --config config.yaml --recover --wall-seconds 28800 \
  >> logs/run-console.log 2>&1
```

Never remove ownership files to force concurrent writers. The nonblocking run
lock prevents overlapping ownership; process identity uses hostname, boot ID and
process start token. Explicit recovery verifies that ownership can be acquired.
Failed/unknown recovery consumes the original persistent retry allowance, chosen
at launch; repeated invocations cannot reset it. Clean safe-stop resumes do not
consume retries. Default allowance is zero. No service, timer, retry loop or
scheduler relaunch is installed. Any automatic recovery policy must be separately
implemented/authorized, opt-in, bounded by the persisted budgets and retries,
and resume a verified checkpoint; this implementation has no such policy.

`recovery.pt` is independent of validation improvement and written every completed
epoch (`runtime.recovery_every=1` is the supported cadence). There is an initial
recovery point before updates. On crash, at most the current uncommitted epoch is
replayed. Model, optimizer, optional scheduler, best-selection state, phase/next
epoch/global updates, Python/NumPy/Torch RNG, standard loader/sampler generator
state and cumulative timing are restored. Data iteration restarts at the epoch
boundary with the saved RNG state. Supported exact ordering uses zero-worker
standard DataLoaders/default collation with TensorDataset, LiTS dataset or Subset
and supported sequential/random samplers; unsupported loader types are refused.
Generator sharing and the ordered parameters of repository preprocessing
transforms (including `torchvision.Compose` pipelines) belong to that identity.
Custom transform callables or extra transform state are refused because their
local RNG state cannot be restored reliably.
Phase B resumes directly and does not replay Phase A or repeat best-A initialization.
The one-time phase transition also has an atomic continuation boundary.

Each checkpoint is serialized to a same-filesystem temporary file, flushed,
fsynced, verified and atomically replaced. `recovery.previous.pt` retains one
verified predecessor; temporary/incomplete snapshots are never advertised as
resumable. If the latest is corrupt, explicit recovery can use the verified prior
snapshot while preserving corruption evidence. Best selection files remain
separate under `phase_a/best.pt` and `phase_b/best.pt`; phase-end `last.pt` remains
available. Resume validates schema, model/configuration and data/split content
identity. Changed cohort, preprocessing, cadence or model settings are refused.
Existing legacy `best.pt`/`last.pt` without continuation state support evaluation
or inference only, not exact resume. Do not overwrite them to synthesize a resume.

## Validation, completion and evaluation

`runtime.validation_every=1` retains full validation every epoch. A larger value
runs full validation on the selected epochs and every phase's final epoch; it
changes best-model selection opportunities and belongs to experiment identity.
Choose it before launching and do not change it during resume.

Training does not automatically evaluate after completion. To include final
Phase B best evaluation in the same allocation, pass `--final-eval`; defaults are
`runtime.final_eval_wall_seconds=300`, `runtime.final_eval_max_batches=100`, and
the remaining training budgets. Incomplete bounded evaluation withholds metrics
rather than presenting a partial cohort as a full-split result. Its stop event
records `training_completed=true` separately from the attempt's stop reason.
The completed continuation checkpoint prevents replaying training updates. Use an
explicit standalone bounded evaluation if more evaluation allocation is needed:

```bash
dense-unet-3d eval --config config.yaml \
  --checkpoint models/example_run/phase_b/best.pt \
  --wall-seconds 600 --max-batches 100
```

Standalone evaluation counts setup/loaders/model loading within its own wall
limit and never changes the training run's budget, ownership or checkpoints.
Evaluation checks its limits between batches; one batch or a blocked data read
can overrun the requested wall time. Never publish partial metrics after a bound
is reached. For handoff, provide config path, run directory, source identity,
last terminal reason/attempt, consumed/remaining budgets, recovery checkpoint
location, latest epoch/phase/update count, and relevant console/event log paths.

## Verification evidence

The CPU tests below cover the implementation contracts. Full commands/results
are reported with the change; test names are pointers, not claims of GPU execution.

| Requirement | CPU test evidence |
| --- | --- |
| Weights/optimizer/scheduler/RNG/order/best/updates equal across resume, both phases | `tests/training/test_recovery.py::test_exact_resume` (boundary and scheduler variants) |
| Transition interruption avoids repeated updates | `test_failed_transition_commit_replays_no_updates` |
| Training/validation/checkpoint signal safe boundary | `test_signals_finish_consistent_boundary` |
| Atomic write/disk failure keeps recovery and predecessor | `test_atomic_write_failure_preserves_latest_and_previous` |
| Plateau recovery independent of best selection | `test_exact_resume`; recovery snapshot assertions each boundary |
| Persistent budget, downtime accounting, checkpoint latency | `test_budget_persists_and_lost_attempt_charges_downtime`, `test_checkpoint_latency_counted_and_no_restart_after_budget` |
| Duplicate ownership/retry allowance | `test_duplicate_owner_and_append_attempts`, `test_persistent_retry_exhaustion` |
| Incompatible/legacy/corrupt checkpoints; verified fallback | `test_incompatible_and_unreadable_recovery_refused`, `test_corrupt_latest_fallback_preserves_evidence` |
| Nonfinite gradients/storage/monitor failures | `test_nonfinite_gradient_keeps_initial_recovery`, `test_storage_threshold_and_background_failure_are_failures` |
| CLI plan before allocation/preflight, persisted budget, forced CPU, terminal distinction | `tests/test_runtime_cli.py` |
| Bounded final eval withholds partial results, records completed training; nonfinite logits | `test_final_evaluation_stop_preserves_training_completion`, `test_bounded_eval_withholds_incomplete_metrics`, `test_bounded_eval_rejects_nonfinite_logits` |
| Existing evaluation/predict/CLI behavior | `tests/test_cli.py`, `tests/evaluation/test_evaluate.py` |

Remaining operational prerequisite: a separately authorized, bounded real GPU
smoke showing checkpoint-and-stop, process exit, resume and completion within the
selected allocation before any multi-day launch. CPU tests do not validate GPU
kernel behavior, real-data throughput, physical host reboot or real disk-full
hardware conditions. No GPU smoke or training launch is part of this change.
