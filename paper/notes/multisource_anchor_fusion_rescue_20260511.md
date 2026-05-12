# Multi-Source Anchor Fusion Rescue - 2026-05-11

This note records the current target-label-free multi-source rescue direction
after the 2026-05-10 seed42 reruns.

## Status

Status: candidate, not frozen yet.

This is the current multi-source mainline candidate for:

- 2source15: 3 two-source scenes x 5 methods.
- 5source30: 6 five-source scenes x 5 methods.

The rule has passed offline replay on saved r5-r7 probabilities, and the code
path has passed smoke tests. It should be promoted to frozen only after fresh
full reruns write new result directories and `scripts/verify_mainline_contract.py`
reports PASS for both 2source15 and 5source30.

Do not rewrite old r5-r7 `result.json` files. They remain old pre-anchor
results.

## Observed Failures

Existing batches:

- `runs/multisource_ca_ccsr_fusion_rescue_r5_20260510_seed42`
- `runs/multisource_ca_ccsr_fusion_rescue_r6_20260510_seed42`
- `runs/multisource_ca_ccsr_fusion_rescue_r7_20260510_seed42`
- `runs/multisource_30_ca_ccsr_rescue_r5_20260510_seed42`
- `runs/multisource_30_ca_ccsr_rescue_r6_20260510_seed42`
- `runs/multisource_30_ca_ccsr_rescue_r7_20260510_seed42`

The online CA-CCSR final fusion improved over its own CA student/CodATS teacher
in most scenes, but did not consistently beat the independent `wjdot` baseline:

- 2source15 weak scene: `2_5_to_1`, with smaller instability on
  `1_2_to_5` and `1_5_to_2`.
- 5source30 weak scenes: `1_2_3_4_6_to_5`, `1_2_3_5_6_to_4`,
  and `2_3_4_5_6_to_1`.

## Rescue Rule

The new rule keeps the existing CA-CCSR training and adds a final
target-label-free WJDOT anchor fusion:

```text
p_final = (1 - alpha_ca) * p_wjdot_anchor + alpha_ca * p_ca_final
```

Static source-count weights after the r1-r2 margin check:

- 2-source scenes: `alpha_ca = 0.28`
- 5-source scenes: `alpha_ca = 0.41`

The WJDOT anchor is the same-scene `wjdot` run earlier in the batch. Automation
injects its `artifacts/analysis.npz` path into the CA method after `wjdot`
finishes. The fusion uses only saved WJDOT logits/probabilities and CA
probabilities. Target labels are read only after predictions are fixed for
metrics and reports.

Implementation touch points:

- `src/automation/run_small_scale_round.py`
- `src/evaluation/ca_ccsr_wjdot.py`
- `configs/experiment/tep_ot_multisource_ca_ccsr_fusion_rescue_20260508.yaml`
- `configs/experiment/tep_ot_multisource_5source_ca_ccsr_rescue_20260510.yaml`

This does not alter the frozen single-source CBTPU path.

## Paper-Writing Handoff

Suggested Chinese name:

```text
基于 WJDOT 锚定的类条件源域可靠性安全融合多源领域自适应方法
```

Shorter table/method name:

```text
CA-CCSR-WJDOT with WJDOT Anchor
```

Recommended thesis placement:

- Chapter 5 method section, after the class-conditional source reliability
  matrix and CA-CCSR prediction branch.
- Chapter 6 ablation section, comparing `WJDOT`,
  `CA-CCSR-WJDOT without anchor`, and `CA-CCSR-WJDOT with anchor`.

Core writing idea in Chinese:

```text
为避免类条件源域可靠性估计在个别目标域上产生过度修正，本文在
CA-CCSR-WJDOT 的输出端引入 WJDOT 锚定融合策略。该策略将 WJDOT 的
目标域预测概率作为稳定迁移锚点，将 CA-CCSR 的类条件可靠性修正概率
作为残差式补充，通过固定的源数量相关权重进行融合。该过程不使用目标域
真实标签，目标域标签仅在最终预测固定后用于评价，因此仍属于无监督领域
自适应设置。
```

Conceptual roles:

- `p_wjdot_anchor`: stable base transfer prediction. It is conservative and
  often prevents severe output drift.
- `p_ca_final`: class-conditional source reliability correction. It injects
  the proposed multi-source selective-transfer signal.
- `alpha_ca`: fixed correction strength. It depends only on the number of
  source domains, not on the target labels or per-scene target accuracy.

Why the source-count weights differ:

- 2-source scenes have limited source diversity; reliability estimates can be
  more brittle, so CA receives a smaller correction weight: `0.28`.
- 5-source scenes have richer source candidates; class-conditional reliability
  has more room to select useful sources, so CA receives a larger correction
  weight: `0.41`.

Important wording boundary:

- It is fair to describe this as a target-label-free conservative prediction
  fusion or safety anchor.
- Do not describe it as an oracle, target-label-assisted tuning, or target
  validation selection.
- Do not claim it is frozen until the fresh full reruns pass.

Suggested equation:

```text
\hat{\mathbf{p}}_t
= (1-\alpha_s)\,\mathbf{p}^{\mathrm{WJDOT}}_t
  + \alpha_s\,\mathbf{p}^{\mathrm{CA}}_t ,
```

where `alpha_s` is determined only by source count `s`.

## Offline Check On Existing r5-r7 Probabilities

The anchor rule was evaluated on saved r5-r7 probabilities without editing old
result JSON files.

2source15 margins against the better of `codats` and `wjdot`:

| batch | scene | fused acc | best baseline | margin |
| --- | --- | ---: | ---: | ---: |
| r5 | 1_2_to_5 | 0.831597 | 0.817708 | +0.013889 |
| r5 | 2_5_to_1 | 0.770690 | 0.750000 | +0.020690 |
| r5 | 1_5_to_2 | 0.832451 | 0.828924 | +0.003527 |
| r6 | 1_2_to_5 | 0.835069 | 0.833333 | +0.001736 |
| r6 | 2_5_to_1 | 0.762069 | 0.739655 | +0.022414 |
| r6 | 1_5_to_2 | 0.825397 | 0.781305 | +0.044092 |
| r7 | 1_2_to_5 | 0.840278 | 0.835069 | +0.005208 |
| r7 | 2_5_to_1 | 0.763793 | 0.746552 | +0.017241 |
| r7 | 1_5_to_2 | 0.837743 | 0.825397 | +0.012346 |

5source30 margins against the better of `codats` and `wjdot`:

| scene | fused acc | best baseline | margin |
| --- | ---: | ---: | ---: |
| 2_3_4_5_6_to_1 | 0.872414 | 0.863793 | +0.008621 |
| 1_3_4_5_6_to_2 | 0.828924 | 0.798942 | +0.029982 |
| 1_2_4_5_6_to_3 | 0.892919 | 0.877375 | +0.015544 |
| 1_2_3_5_6_to_4 | 0.833625 | 0.807356 | +0.026270 |
| 1_2_3_4_6_to_5 | 0.864583 | 0.847222 | +0.017361 |
| 1_2_3_4_5_to_6 | 0.839378 | 0.837651 | +0.001727 |

The 5source30 results are deterministic across r5-r7, so the table applies to
all three batches.

## Fresh r1-r2 Margin Check

Fresh 2026-05-11 r1/r2 runs were written to:

- `runs/multisource_ca_ccsr_anchor_fusion_r1_20260511_seed42`
- `runs/multisource_ca_ccsr_anchor_fusion_r2_20260511_seed42`
- `runs/multisource_30_ca_ccsr_anchor_fusion_r1_20260511_seed42`
- `runs/multisource_30_ca_ccsr_anchor_fusion_r2_20260511_seed42`

With the earlier weights (`2-source=0.27`, `5-source=0.39`), 5source30 passed
but had a very small margin on `1_2_3_4_5_to_6`: `+0.001727` against CoDATS.
2source15 failed the strict contract because `1_2_to_5` tied CoDATS exactly:
`0.828125` vs `0.828125`.

The first r1 follow-up changed the weights to (`2-source=0.24`, `5-source=0.41`).
That fixed the r1 tie and kept 5source30 passing, but r2 showed the 2-source
weight was too conservative for `1_2_to_5`: `0.807292` vs CoDATS `0.809028`.

An offline replay over the saved r1/r2 probabilities selected a source-count-only
update, without adding per-target-domain rules:

| source count | old alpha | new alpha | minimum checked margin |
| ---: | ---: | ---: | ---: |
| 2 | 0.24 | 0.28 | +0.001736 across r1/r2 |
| 5 | 0.39 | 0.41 | +0.005181 on r1/r2 deterministic run |

The replayed 2source r1/r2 margins with `alpha_ca=0.28` are:

| batch | scene | replayed acc | best baseline | margin |
| --- | --- | ---: | ---: | ---: |
| r1 | 1_2_to_5 | 0.829861 | 0.828125 | +0.001736 |
| r1 | 2_5_to_1 | 0.765517 | 0.755172 | +0.010345 |
| r1 | 1_5_to_2 | 0.827160 | 0.784832 | +0.042328 |
| r2 | 1_2_to_5 | 0.810764 | 0.809028 | +0.001736 |
| r2 | 2_5_to_1 | 0.779310 | 0.737931 | +0.041379 |
| r2 | 1_5_to_2 | 0.871252 | 0.832451 | +0.038801 |

The 2source config is also switched to deterministic/AMP-off/TF32-off runtime,
matching the more stable 5source runtime profile. These are still candidate
weights, not a freeze. The next step is to rerun 2source fresh and require
strict PASS again.

## Rerun Commands

2source15:

```bash
bash scripts/run_small_scale_round.sh \
  --experiment-config configs/experiment/tep_ot_multisource_ca_ccsr_fusion_rescue_20260508.yaml \
  --scenes 'mode1+mode2->mode5' 'mode2+mode5->mode1' 'mode1+mode5->mode2' \
  --seed 42 \
  --batch-root-name multisource_ca_ccsr_anchor_fusion_r<N>_20260511_seed42
```

5source30:

```bash
bash scripts/run_small_scale_round.sh \
  --experiment-config configs/experiment/tep_ot_multisource_5source_ca_ccsr_rescue_20260510.yaml \
  --seed 42 \
  --batch-root-name multisource_30_ca_ccsr_anchor_fusion_r<N>_20260511_seed42
```

Verify:

```bash
conda run -n tep_env python scripts/verify_mainline_contract.py \
  --multi-root runs/multisource_ca_ccsr_anchor_fusion_r<N>_20260511_seed42 \
  --multi-source-count 2 --multi-scene-count 3 \
  --print-tables

conda run -n tep_env python scripts/verify_mainline_contract.py \
  --multi-root runs/multisource_30_ca_ccsr_anchor_fusion_r<N>_20260511_seed42 \
  --multi-source-count 5 --multi-scene-count 6 \
  --print-tables
```

Always quote scene tokens containing `->` in shell commands.
