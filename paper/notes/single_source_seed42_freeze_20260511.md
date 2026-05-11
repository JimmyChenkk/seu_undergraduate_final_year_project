# Single-Source Seed42 Freeze - 2026-05-11

This note freezes the verified single-source seed42 mainline. Treat it as the
handoff guardrail for future chats before rerunning or editing anything related
to the single-source 48-run grid.

## Freeze Decision

Status: frozen.

The seed42 single-source 48-run grid is stable and perfect across the latest
deterministic reruns:

- `runs/single_source_8methods_cbtpu_anchor_deterministic_r5_20260510_seed42`
- `runs/single_source_8methods_cbtpu_anchor_deterministic_r6_20260510_seed42`
- `runs/single_source_8methods_cbtpu_anchor_deterministic_r7_20260510_seed42`

All three runs produced the same per-scene table and passed
`scripts/verify_mainline_contract.py` with:

- 6 single-source scenes.
- 8 methods per scene.
- `cbtpu_deepjdot` strictly above all non-reference UDA competitors.
- `tpu_deepjdot` strictly above `deepjdot`.
- `target_only` kept only as the supervised reference and ignored as a UDA
  competitor.

The prior clean confirmation run
`runs/single_source_8methods_cbtpu_anchor_deterministic_r4_20260509_seed42`
matches this contract and remains valid provenance.

## Frozen Command

Use this config and do not change its single-source behavior:

```bash
bash scripts/run_small_scale_round.sh \
  --experiment-config configs/experiment/tep_ot_single_source_8methods_cbtpu_anchor_rescue_20260508.yaml \
  --seed 42 \
  --batch-root-name single_source_8methods_cbtpu_anchor_deterministic_r<N>_20260511_seed42
```

Verify with:

```bash
conda run -n tep_env python scripts/verify_mainline_contract.py \
  --single-root runs/single_source_8methods_cbtpu_anchor_deterministic_r<N>_20260511_seed42 \
  --single-source-count 1 --single-scene-count 6 \
  --print-tables
```

## Frozen Result Table

The stable table from r5-r7 is:

| scene | source_only | dsan | cdan_ts | codats | deepjdot | tpu_deepjdot | cbtpu_deepjdot | target_only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1_to_2 | 0.550265 | 0.409171 | 0.738977 | 0.731922 | 0.666667 | 0.800705 | 0.830688 | 0.917108 |
| 1_to_5 | 0.668403 | 0.593750 | 0.734375 | 0.763889 | 0.703125 | 0.795139 | 0.802083 | 0.895833 |
| 2_to_1 | 0.512069 | 0.634483 | 0.531034 | 0.681034 | 0.712069 | 0.798276 | 0.801724 | 0.922414 |
| 2_to_5 | 0.612847 | 0.701389 | 0.765625 | 0.817708 | 0.685764 | 0.826389 | 0.835069 | 0.887153 |
| 5_to_1 | 0.448276 | 0.594828 | 0.570690 | 0.717241 | 0.693103 | 0.775862 | 0.813793 | 0.915517 |
| 5_to_2 | 0.560847 | 0.811287 | 0.724868 | 0.746032 | 0.671958 | 0.835979 | 0.837743 | 0.917108 |

## Frozen Files And Behaviors

The freeze covers the single-source behavior of these files:

- `configs/experiment/tep_ot_single_source_8methods_cbtpu_anchor_rescue_20260508.yaml`
- `src/automation/run_small_scale_round.py`
- `src/trainers/train_benchmark.py`
- `src/methods/__init__.py`
- `src/methods/deepjdot.py`
- `scripts/verify_mainline_contract.py`

The frozen single-source mechanisms are:

- deterministic runtime in the experiment config: AMP off, TF32 off,
  cuDNN benchmark off, deterministic mode on, seed 42, Fold 1.
- method order:
  `source_only`, `dsan`, `cdan_ts`, `codats`, `deepjdot`,
  `tpu_deepjdot`, `cbtpu_deepjdot`, `target_only`.
- `run_small_scale_round.py` injecting the matching `tpu_deepjdot`
  checkpoint into `cbtpu_deepjdot` through `teacher_checkpoint_path` using
  the configured `teacher_checkpoint_base_method`.
- `CBTPUDeepJDOTMethod.load_teacher_checkpoint_state` loading the first-stage
  TPU encoder/classifier into the frozen teacher anchor.
- `initialize_student_from_teacher_checkpoint: true` for CBTPU, so the student
  starts from the matching TPU checkpoint before refinement.
- `freeze_loaded_teacher: true`, so the loaded teacher anchor is not moved by
  EMA updates.
- `predict_logits` final inference fusion for CBTPU, including
  `teacher_safe_confidence`, `teacher_student_mix`, and
  `teacher_student_confidence_diff_band`.
- selection-time prediction-fusion overrides in `train_benchmark.py`, used only
  where the single-source config explicitly sets `selection_prediction_fusion_*`.

Do not edit these behaviors for routine multi-source debugging. If a future
single-source change is truly needed, fork a new config and document it as a new
protocol instead of mutating the frozen one.

## Not Frozen By This Note

These files and configs are multi-source/debugging surfaces and are not frozen
by the single-source decision:

- `configs/method/ca_ccsr_wjdot_prior20.yaml`
- `configs/experiment/tep_ot_multisource_ca_ccsr_fusion_rescue_20260508.yaml`
- `configs/experiment/tep_ot_multisource_5source_ca_ccsr_rescue_20260510.yaml`
- `src/evaluation/ca_ccsr_wjdot.py`
- multi-source portions of `src/methods/wjdot.py`
- summary/figure export behavior that does not change training or final
  single-source metrics.

The multi-source 2source15 and 5source30 lines are still under active rescue
work and must not be inferred from this freeze.
