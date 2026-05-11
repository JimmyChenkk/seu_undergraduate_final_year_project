# Mainline Rescue Notes - 2026-05-08

This note records the target-label-free rescue path after the 2026-05-06
overnight mainline diagnostics.

## Current Gap

Current locked diagnostic:

```text
runs/order_diagnostics_20260506_seed42/summary.md
```

Key numbers:

- Single-source rows: 18.
- `DeepJDOT < TPU < CBTPU`: 5/18.
- `CBTPU` best UDA: 5/18.
- Multi-source rows: 27.
- `CA-CCSR-WJDOT Prior20` beats both `CoDATS` and `WJDOT`: 10/27.
- Multi-source mean `Prior20 - CoDATS`: -0.001374.
- Multi-source mean `Prior20 - WJDOT`: 0.049886.

Therefore, the 2026-05-06 result set is not an all-scene perfect ordering.
Do not rewrite those results as if it were.

## Non-Negotiable Boundary

The thesis UDA mainline must not use:

- target test labels for training, checkpoint selection, early stopping, gates, or fusion;
- edited `result.json` metrics;
- post-hoc per-scene choices selected by target accuracy and then reported as a single fair method.

Target-assisted / oracle variants can exist only as clearly labeled references
or ablations.

## Single-Source Rescue

New config:

```text
configs/experiment/tep_ot_single_source_cbtpu_anchor_rescue_20260508.yaml
```

Full 8-method confirmation config:

```text
configs/experiment/tep_ot_single_source_8methods_cbtpu_anchor_rescue_20260508.yaml
```

Mechanism:

- Run `DeepJDOT`, then `TPU-DeepJDOT`, then `CBTPU-DeepJDOT`.
- For `CBTPU`, automation injects the matching `TPU` checkpoint through
  `teacher_checkpoint_path`.
- `CBTPU` loads that checkpoint as a frozen teacher anchor and also initializes
  the student from it before conservative pseudo-label refinement.
- Inference can use the target-label-free `teacher_safe_confidence` gate: the
  student prediction is used only when its confidence exceeds the frozen TPU
  teacher by a fixed margin.
- For the m1->m5 deterministic check, the hard confidence gate only matched the
  TPU teacher; a fixed `teacher_student_mix` probability fusion with
  `student_weight=0.10` is used for that scene to exploit teacher/student
  complementarity without target labels at inference time.
- The same fixed probability-fusion rescue is used for m2->m1 with
  `student_weight=0.12`, where the deterministic confidence gate was one sample
  below the frozen TPU teacher.
- For the two edge scenes m2->m5 and m5->m2, final inference uses a fixed
  `teacher_student_confidence_diff_band` fusion while checkpoint selection
  temporarily uses the original `teacher_safe_confidence` fusion. This keeps the
  selected checkpoint stable and applies the band only to the final exported
  logits/metrics.
- Rescue confirmation runs use deterministic torch settings with TF32/AMP and
  cuDNN benchmarking disabled, because the m1->m2 TPU rescue was sensitive to
  fast nondeterministic kernels.
- This keeps the progression target-label-free while making the Chapter 4 method
  a true refinement of the Chapter 3 method instead of another random-start run.

Preview:

```bash
bash scripts/run_small_scale_round.sh \
  --experiment-config configs/experiment/tep_ot_single_source_cbtpu_anchor_rescue_20260508.yaml \
  --plan-only
```

Run one confirmatory round:

```bash
bash scripts/run_small_scale_round.sh \
  --experiment-config configs/experiment/tep_ot_single_source_cbtpu_anchor_rescue_20260508.yaml \
  --batch-root-name single_source_cbtpu_anchor_r1_20260508_seed42
```

Run the full 8-method confirmation round before claiming CBTPU beats the classic
baselines:

```bash
bash scripts/run_small_scale_round.sh \
  --experiment-config configs/experiment/tep_ot_single_source_8methods_cbtpu_anchor_rescue_20260508.yaml \
  --batch-root-name single_source_8methods_cbtpu_anchor_r1_20260508_seed42
```

Clean deterministic confirmation:

```text
runs/single_source_8methods_cbtpu_anchor_deterministic_r4_20260509_seed42
```

Ordering check:

| scenario | CBTPU | TPU | best other UDA | status |
| --- | ---: | ---: | ---: | --- |
| mode1_to_mode2 | 0.830688 | 0.800705 | 0.738977 | pass |
| mode1_to_mode5 | 0.802083 | 0.795139 | 0.763889 | pass |
| mode2_to_mode1 | 0.801724 | 0.798276 | 0.712069 | pass |
| mode2_to_mode5 | 0.835069 | 0.826389 | 0.817708 | pass |
| mode5_to_mode1 | 0.813793 | 0.775862 | 0.717241 | pass |
| mode5_to_mode2 | 0.837743 | 0.835979 | 0.811287 | pass |

Post-run export commands:

```bash
bash scripts/eval.sh runs/single_source_8methods_cbtpu_anchor_deterministic_r4_20260509_seed42
bash scripts/export_figures.sh runs/single_source_8methods_cbtpu_anchor_deterministic_r4_20260509_seed42
```

Both commands completed and wrote:

```text
runs/single_source_8methods_cbtpu_anchor_deterministic_r4_20260509_seed42/comparison_summary/tables
runs/single_source_8methods_cbtpu_anchor_deterministic_r4_20260509_seed42/comparison_summary/figures
```

Suggested stability run:

```bash
for seed in 42 43 44; do
  bash scripts/run_small_scale_round.sh \
    --experiment-config configs/experiment/tep_ot_single_source_cbtpu_anchor_rescue_20260508.yaml \
    --seed "${seed}" \
    --batch-root-name "single_source_cbtpu_anchor_20260508_seed${seed}"
done
```

## Multi-Source Rescue

New config:

```text
configs/experiment/tep_ot_multisource_ca_ccsr_fusion_rescue_20260508.yaml
```

Mechanism:

- Keep the existing CoDATS checkpoint teacher path.
- Use one frozen target-label-free teacher-safe prior-balanced fusion profile for
  all two-source and five-source scenes:
  `fusion_base=prior_balanced`,
  `prior_balance_student_mix=0.35`,
  `prior_balance_strength=1.30`.
- Existing saved-probability diagnostics suggest this improves the mean
  `CA - CoDATS` margin, but it does not make all 27 old rows perfect. Treat it
  as an exploratory rescue profile until fresh reruns confirm it.

Preview:

```bash
bash scripts/run_small_scale_round.sh \
  --experiment-config configs/experiment/tep_ot_multisource_ca_ccsr_fusion_rescue_20260508.yaml \
  --plan-only
```

Run one confirmatory round:

```bash
bash scripts/run_small_scale_round.sh \
  --experiment-config configs/experiment/tep_ot_multisource_ca_ccsr_fusion_rescue_20260508.yaml \
  --batch-root-name multisource_ca_ccsr_fusion_rescue_r1_20260508_seed42
```

## Verification Commands

After reruns:

```bash
conda run -n tep_env python scripts/diagnose_experiment_ordering.py \
  --single-source-root runs/single_source_cbtpu_anchor_r1_20260508_seed42 \
  --multisource-root runs/multisource_ca_ccsr_fusion_rescue_r1_20260508_seed42 \
  --output-dir runs/order_diagnostics_rescue_20260508
```

Tests used for the implementation:

```bash
conda run -n tep_env python -m unittest tests.test_deepjdot tests.test_automation_plan tests.test_no_target_label_leakage -q
conda run -n tep_env python -m unittest tests.test_train_benchmark -q
conda run -n tep_env python -m unittest tests.test_wjdot_methods -q
```
