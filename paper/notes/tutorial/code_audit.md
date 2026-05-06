# TEP OT Method Audit

Date: 2026-05-02

## Scope

This audit covers the local WJDOT family implementation in:

- `src/tep_ot/ot_losses.py`
- `src/methods/wjdot.py`
- `src/tep_ot/methods.py`
- `src/trainers/selection_metrics.py`
- `src/trainers/train_benchmark.py`
- `configs/method/{wjdot,tp_wjdot,cbtp_wjdot,ms_cbtp_wjdot}.yaml`
- `configs/experiment/tep_ot_methods_9tasks_fixedfold.yaml`

Target labels are hidden in the DA training loaders and are only used by final
evaluation/export paths.

## Current Implementation

WJDOT and the TP/CBTP/MS variants are implemented twice:

- Benchmark trainer path: `src/methods/wjdot.py`, built by `src/methods/__init__.py`.
- Focused TEP OT runner path: `src/tep_ot/methods.py`, built by `src/tep_ot/methods.py::build_method`.

The shared OT loss is `src/tep_ot/ot_losses.py::jdot_transport_loss`.

Prototype cost currently supports `prototype_in_coupling=false`. In that case,
the OT plan is solved from feature + label costs, but the prototype cost is
still added afterward under the fixed coupling:

```text
gamma = OT(base_feature_cost + base_label_cost)
loss = sum gamma * (base_cost + prototype_weight * pairwise_prototype_cost)
```

This is safer than putting prototypes directly into the transport plan, but it
is still a pairwise source-label pull: each target sample receives prototype
forces through all source samples/classes that the plan assigns mass to.

Confidence curriculum, class-balanced pseudo-label CE, and consistency are in:

- `src/methods/wjdot.py::_target_regularizers`
- `src/tep_ot/methods.py::JDOTMethod._target_regularizers`

CBTP enables those regularizers by setting `confidence_curriculum=true`. MS-CBTP
adds source-level, class-level, and sample-level weighting. In the benchmark
path, single-source MS-CBTP already calls the CBTP path when
`len(source_batches) == 1`, but it does not yet emit an explicit diagnostic.

Checkpoint selection is target-label-free when configs use
`hybrid_source_eval_confidence_guard`; however, the trainer still records
`target_eval_acc` periodically for monitoring. The selection score itself uses
source validation and target prediction proxies, not target labels.

## Current Parameters

`configs/experiment/tep_ot_methods_9tasks_fixedfold.yaml` currently sets:

- WJDOT: `adaptation_weight=0.35`, `feature_weight=0.03`,
  `alignment_start_step=350`, `alignment_ramp_steps=700`.
- TP-WJDOT: `prototype_weight=0.01`, `prototype_in_coupling=false`,
  delayed prototype start/ramp, dropout `0.2`, weight decay `1e-4`,
  confidence-ceiling checkpoint guard.
- CBTP/MS: `prototype_weight=0.08`, `prototype_in_coupling=false`,
  `pseudo_weight=0.06`, `consistency_weight=0.04`,
  confidence-ceiling checkpoint guard.

Method configs mirror those defaults in `configs/method/tp_wjdot.yaml`,
`configs/method/cbtp_wjdot.yaml`, and `configs/method/ms_cbtp_wjdot.yaml`.

## Log Review

Recent local comparison summaries show the main pattern:

- `runs/wjdot_repair_probe_m1_m5_v9`: WJDOT selected target accuracy `0.7222`;
  TP `0.6979`; CBTP/MS `0.7240`.
- `runs/wjdot_repair_probe_m1_m5_v6`: TP `0.6858`; CBTP `0.7257`;
  MS `0.7240`.
- `runs/wjdot_repair_probe_m1_m2`: WJDOT `0.7231`; TP `0.6684`;
  CBTP `0.6772`; MS `0.6861`.

The pattern supports the interpretation that temporal prototypes alone are not
reliably beneficial, while confidence-balanced curriculum and consistency make
the prototype signal usable.

## Why TP Can Over-Align

1. Pairwise hard pull remains even with `prototype_in_coupling=false`.
   The plan is fixed before adding prototype cost, but the residual still
   penalizes target samples according to source sample labels. Early noisy
   transport mass can pull target embeddings toward the wrong source class
   prototype.

2. Prototype gating in TP can use classifier confidence through
   `prototype_confidence_threshold`. That is not target-label leakage, but it
   makes TP closer to CBTP and can amplify early over-confidence.

3. The residual prototype loss does not currently use OT-induced class
   uncertainty. A target sample with diffuse class mass under the OT plan can
   receive the same prototype force as a sample with a clear transport-induced
   class assignment.

4. Late-epoch over-alignment is plausible because the current residual keeps
   applying pairwise attraction after classifier/source validation has already
   saturated. This can reduce target class diversity even when source metrics
   improve.

## Recommended Code Direction

Keep WJDOT as the baseline, make TP explicitly prototype-regularized rather
than pseudo-label-driven, and keep CBTP/MS as the variants that add classifier
confidence curriculum and consistency. The most important TP mode should be
OT-plan barycentric alignment: use the transport plan to form target class
barycenters and align those class-level barycenters to source prototypes.

