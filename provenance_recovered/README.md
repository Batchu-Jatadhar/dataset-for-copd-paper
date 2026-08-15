# Recovered Provenance for `results_final_study/`

Four scripts that produced `results_final_study/` are absent from the working
tree. This folder holds everything that could be recovered about them. It is
**forensic evidence, not source code** — nothing here is runnable, and nothing
here was reimplemented, reconstructed, or inferred.

## Origin

All four scripts lived in `Emphysema_Detection/` and were written and executed on
**13 June** by an Antigravity IDE (Gemini) agent session:

```
C:\Users\batch_dcahfiw\.gemini\antigravity-ide\brain\bd0d098a-e138-460c-b5a0-859081a951f3
```

Source transcript `.system_generated/logs/transcript.jsonl`
(674,865 bytes, sha256 `88083b06506442893482bc5e0957de9adeebad4c831f711320112b7dec1e90d3`).

| Description | Actual filename | Produced | Run finished |
|---|---|---|---|
| adversarial audit | `adversarial_audit.py` | `results_final_study/adversarial_audit/` | 13:50 |
| shortcut isolation | `shortcut_isolator.py` (v2) | `results_final_study/shortcut_isolation/` | 14:39 |
| corrected run | `phase_corrected.py` | `results_final_study/corrected/` | 15:29 |
| preprocessing ablation | `phase_ablation.py` (v2) | `results_final_study/ablation/` | 19:07 |

## Recovery status

**The four scripts are not recoverable as working code.** The original files do
not exist anywhere on this machine, and no complete copy was ever stored.

The transcript logged each `write_to_file` call, but its logger capped the
recorded content at roughly 1.8 KB and appended a literal `<truncated N bytes>`
marker. The truncated remainder was never written to disk in any form. What
survives is a verbatim prefix of each script.

| File | Recovered | Original | Coverage |
|---|---|---|---|
| `adversarial_audit.py.fragment` | 1,761 B | ~39,187 B | 4.5% |
| `shortcut_isolator.py.fragment` | 1,780 B | ~24,839 B | 7.2% |
| `phase_corrected.py.fragment` | 1,748 B | ~28,086 B | 6.2% |
| `phase_ablation.py.fragment` | 1,751 B | ~16,117 B | 10.9% |
| `_test_pid.py` | 1,115 B | 1,115 B | **100%** |

Each fragment contains the complete module docstring (which states the
experimental design), the full import block, and the beginning of the CONFIG
block. That is enough to document methodology. It is **not** enough to re-run
anything.

`_test_pid.py` is the exception: it was small enough to escape truncation and is
recovered **in full**. It is the diagnostic that established the ChestX6
patient-ID structure, and it contains the exact `N_<num>` / `E_<num>` regex that
`phase_corrected.py`'s docstring says it adopted as the patient-ID fix.

### Fidelity

The decoder was validated against `Emphysema_Detection/train_final_baseline.py`,
which was captured by the same logging path but still exists on disk. The
recovered fragment matches that file byte-for-byte for its first 584 characters
and then diverges at exactly one point — an `import sys` line that a *later* edit
added to the file. The fragments are faithful verbatim prefixes, not
reconstructions.

Everything above the truncation banner in each `.fragment` file is verbatim. The
banner itself is the only text added during recovery.

## Contents

### `code_fragments/`
- `adversarial_audit.py.fragment`, `shortcut_isolator.py.fragment`,
  `phase_corrected.py.fragment`, `phase_ablation.py.fragment` — verbatim
  prefixes. The `.fragment` extension is deliberate: these must never be
  imported or executed.
- `_test_pid.py` — fully recovered, complete and coherent.
- `surviving_edits.md` — two `replace_file_content` calls small enough to be
  logged in full: a `bool()` cast at `adversarial_audit.py:386` and a
  `set_seed()` insertion at `phase_ablation.py:59-60`.

### `run_logs/`
Full stdout of all four runs, plus five supporting session logs. **For
reconstructing methodology these are more complete than the code fragments** —
they carry per-seed metrics, per-epoch traces, split counts, and the assertions
that were checked at runtime.

| File | Contents |
|---|---|
| `task-20_adversarial_audit.log` | All six audit checks, per-seed timings and metrics |
| `task-73_shortcut_isolator.log` | ChestX6 patient-ID root-cause analysis, 2×2 experiment |
| `task-91_phase_corrected.log` | Split verification, per-epoch validation AUC for experiments A–D |
| `task-203_phase_ablation.log` | Per-epoch disease AUC and domain accuracy for all four methods |

### `reports/`
Seven reports written by the same session, including
`adversarial_audit_report.md`, `shortcut_root_cause_report.md`,
`corrected_pipeline_report.md`, and `emphysema_final_verification_report.md`.

### `cleanup_records/`
`cleanup_plan.txt`, `repository_inventory.txt`, and `backup_manifest.txt`,
recovered from a 6 July duplicate of this repo at
`Downloads\research paper\research paper 2\`. They are absent from the current
working tree.

Note that `cleanup_plan.txt` lists a `preprocessing_ablation.py` deleted at 03:45
on 13 June. That is **a different, earlier script** — `phase_ablation.py` did not
produce the ablation results until 19:07 the same day. Do not conflate them.

## Where recovery was attempted and failed

- **Git, both clones** — `main` holds 2 commits; `results_final_study/` was never
  tracked. `--diff-filter=D` shows only `phase1_2_baseline.py`, `phase3_dann.py`
  and old PNGs. Reflog holds nothing beyond those 2 commits, no stashes,
  `git fsck` reports zero dangling or unreachable objects. `main` and
  `origin/main` are identical.
- **`backup_before_cleanup/`** — model weights and PNGs only, no `.py`.
- **`__pycache__/`** — no `.pyc` for any of the four; caches date from 29 July
  and August, the scripts from 13 June.
- **Duplicate repo copy** (6 July snapshot) — does not contain them.
- **Filesystem-wide** — no file by any of these names anywhere on C:, the only
  drive, including OneDrive. No `.py.bak`, `.orig`, `~`, or `.save` files. No
  `.idea` or `.vscode` in either copy. JetBrains `LocalHistory` exists but holds
  nothing for this project. VS Code `Backups` is empty.
- **Recycle Bin** — every `$I` metadata record's original path was scanned; no
  match.

## Citing this in the paper

The fragments establish each script's stated design and dependencies. The run
logs establish the numbers actually produced. Neither establishes the
implementation between those two points, so any methods section describing the
interior of these scripts is describing something that cannot currently be
verified against source. Treat that gap as a known limitation rather than
papering over it.
