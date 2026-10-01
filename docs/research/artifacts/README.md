# Archived diagnostic source

These immutable text snapshots preserve the exact source used by the two
update-bearing GPU attempts and their shared configuration. Their extensions
prevent test, lint and packaging tools from treating historical code as the
current executable implementation. The
[GPU diagnostic receipt](../2026-09-30-gpu-diagnostics.json) preserves hashes,
but not source text, for the two zero-update failures and the later
reporting-only script.

| Artifact | SHA256 |
| --- | --- |
| [executed-diagnostics.py.txt](executed-diagnostics.py.txt) (parity attempt, 10 updates) | `f7e14094024fc4b4e2788675f5c23aa771058871f5a8dcbcdb5ccbbc071d59d9` |
| [executed-overfit-diagnostics.py.txt](executed-overfit-diagnostics.py.txt) (final 80 updates) | `3e226f1de4d9d315e4d1923ac3cabc23d0b77655c5baff80ec123df0edd6807a` |
| [executed-config.yaml.txt](executed-config.yaml.txt) | `88e0cb2bbef07e91aef2640238486ccf9d2f169b715b69147b242c7afc81b934` |

To inspect or reproduce the historical source in an authorized environment,
copy a snapshot to a temporary `.py` or `.yaml` path and verify its SHA256 first.
The scripts are evidence snapshots, not supported entry points. The source text
for hashes `21d5bb5b…`, `2158a012…`, and `5325adfc…` was not retained in the
repository; those ran zero updates, zero updates, and reporting only,
respectively. Large cached
slabs, per-step tensors, checkpoints and logs remain local-only because they may
contain derived patient data and are unnecessary for normal repository use.
