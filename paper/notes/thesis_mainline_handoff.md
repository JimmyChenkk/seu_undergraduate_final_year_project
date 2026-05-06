# Thesis Mainline Handoff

This note is the compact handoff for thesis writing. Treat it as the source of truth for the current paper mainline.

## Current Thesis Title

基于领域自适应的工业过程故障诊断方法研究

## Main Experiments

### Single-source 48-run grid

Use this for Chapter 3, Chapter 4, and single-source parts of Chapter 6.

```bash
bash scripts/run_small_scale_round.sh \
  --data-config configs/data/te_da.yaml \
  --experiment-config configs/experiment/tep_ot_single_source_8methods_stage1_fold0.yaml \
  --batch-root-name single_source_48_seed42_$(date +%Y%m%d_%H%M%S)
```

Experiment config:

- `configs/experiment/tep_ot_single_source_8methods_stage1_fold0.yaml`

Methods in thesis/table order:

- `source_only`
- `dsan`
- `cdan_ts`
- `codats`
- `deepjdot`
- `tpu_deepjdot`
- `cbtpu_deepjdot`
- `target_only`

Core method configs:

- `configs/method/tpu_deepjdot.yaml`
- `configs/method/cbtpu_deepjdot.yaml`
- `configs/method/deepjdot.yaml`
- `configs/method/codats.yaml`
- `configs/method/dsan.yaml`
- `configs/method/cdan_ts.yaml`
- `configs/method/source_only.yaml`
- `configs/method/target_only.yaml`

Core implementation files:

- `src/methods/deepjdot.py`
- `src/losses/domain.py`
- `src/tep_ot/ot_losses.py`
- `src/backbones/fcn.py`
- `src/trainers/train_benchmark.py`
- `src/automation/run_small_scale_round.py`

Chapter 3 method:

- Code name: `tpu_deepjdot`
- Suggested Chinese name: 基于时序原型非平衡联合分布对齐的单源领域自适应故障诊断方法
- Key mechanisms: source supervised CE, warmup supervised contrastive learning, source EMA class prototypes, temporal cost shaping, unbalanced Sinkhorn transport, DeepJDOT-style joint distribution alignment.
- Important note: this stage intentionally does not use target pseudo labels, target labels, or consistency training.

Chapter 4 method:

- Code name: `cbtpu_deepjdot`
- Suggested Chinese name: 基于多证据置信均衡伪标签的单源领域自适应故障诊断方法
- Key mechanisms: inherits TPU-DeepJDOT, EMA teacher, weak/strong time-series augmentation, three-way semantic evidence from OT / teacher prediction / source prototypes, agreement or low-JS pseudo-label screening, confidence thresholding, capped acceptance, logit adjustment, consistency learning.

## Multi-source 45-run grid

Use this for Chapter 5 and multi-source parts of Chapter 6.

```bash
bash scripts/run_small_scale_round.sh \
  --data-config configs/data/te_da.yaml \
  --experiment-config configs/experiment/tep_ot_multisource_ca_ccsr_wjdot_stage1_probe_fold0.yaml \
  --batch-root-name multisource_45_ca_seed42_$(date +%Y%m%d_%H%M%S)
```

Experiment config:

- `configs/experiment/tep_ot_multisource_ca_ccsr_wjdot_stage1_probe_fold0.yaml`

Methods in thesis/table order:

- `source_only`
- `codats`
- `wjdot`
- `ca_ccsr_wjdot_prior20`
- `target_ref`

Core method configs:

- `configs/method/ca_ccsr_wjdot_prior20.yaml`
- `configs/method/ca_ccsr_wjdot.yaml`
- `configs/method/wjdot.yaml`
- `configs/method/codats.yaml`
- `configs/method/source_only.yaml`
- `configs/method/target_ref.yaml`

Core implementation files:

- `src/methods/wjdot.py`
- `src/evaluation/ca_ccsr_wjdot.py`
- `src/evaluation/ccsr_wjdot_fusion.py`
- `src/tep_ot/ot_losses.py`
- `src/losses/domain.py`
- `src/automation/run_small_scale_round.py`

Chapter 5 method:

- Code name: `ca_ccsr_wjdot_prior20` / `ca_ccsr_wjdot`
- Suggested Chinese name: 基于类条件源域可靠性加权的多源领域自适应故障诊断方法
- Key mechanisms: multi-source WJDOT, per-source joint distribution alignment, class-conditional source reliability weights, CoDATS-style adversarial feature constraint, teacher anchor / teacher-safe fusion in the CA branch.
- Important note: `target_ref` is a supervised target-domain reference, not a UDA method.

## Thesis Structure Status

Current LaTeX skeleton is in:

- `paper/thesis.tex`
- `paper/seuthesis-2022.cls`

The title skeleton is already reset. Chapters 3, 4, and 5 should be written from code and experiment configs, not from generic literature prose.

## Ignore / Do Not Let These Distract Writing

These are older or side branches. Keep them available for history/tests unless explicitly archived, but do not use them as the thesis mainline:

- `src/methods/tc_cdan.py`
- `src/methods/rpl_tc_cdan.py`
- `src/methods/ccs_rpl_tc_cdan.py`
- `src/methods/rcta.py`
- `scripts/run_rcta_mode125_ablation.sh`
- `scripts/summarize_rcta_ablation.py`
- `paper/notes/rcta_mode125_ablation_execution.md`
- `paper/notes/rcta_paper_consolidation_strategy.md`
- `paper/notes/thesis_three_innovations_implementation_summary.md`

Related tests may still exist because they protect historical code paths. They should not be used to infer the current thesis story unless the user explicitly revives that branch.

## Clean-up Policy

- Safe clean-up already done: Python `__pycache__` and `*.pyc` generated files were removed.
- Do not delete source files or configs before a final archive/commit decision; several old files are still useful as baselines, tests, or provenance.
- For new chat, read this note first, then inspect only the files listed under the main experiments unless a concrete issue requires broader search.
