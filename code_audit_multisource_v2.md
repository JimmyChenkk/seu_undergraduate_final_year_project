# Multi-Source WJDOT / CCSR Code Audit V2

This audit stays inside the current workspace.

## Files Read

- `src/datasets/te_torch_dataset.py`
- `src/trainers/train_benchmark.py`
- `src/methods/base.py`
- `src/methods/wjdot.py`
- `src/methods/source_only.py`
- `src/methods/codats.py`
- `src/methods/target_only.py`
- `src/methods/__init__.py`
- `src/evaluation/ccsr_wjdot_fusion.py`
- `src/evaluation/ccsr_wjdot_posthoc.py`
- `src/evaluation/evaluate.py`
- `src/evaluation/report_figures.py`
- `configs/method/wjdot.yaml`
- `configs/method/ccsr_wjdot_fusion.yaml`
- `configs/method/codats.yaml`
- `configs/method/target_only.yaml`
- `configs/experiment/tep_ot_multisource_ccsr_wjdot_stage1_fold0.yaml`
- `src/automation/run_small_scale_round.py`

## Answers

1. **Current WJDOT pools source minibatches.**
   `WJDOTMethod.compute_loss()` calls `self.merge_source_batches(source_batches)`, and `SingleSourceMethodBase.merge_source_batches()` concatenates all source `x/y` tensors with `torch.cat`. The method then computes one source CE, one target prediction, one OT cost matrix, and one OT plan over the pooled source batch.

2. **Current plain WJDOT does not preserve source identity during training.**
   The dataloader keeps separate loaders per source domain, but plain `wjdot` discards identity when it concatenates the minibatches. Downstream WJDOT metrics therefore cannot distinguish which source produced which transported mass.

3. **Current plain WJDOT cannot obtain per-source OT plan `gamma_k`.**
   `jdot_transport_loss()` solves one coupling for the pooled source batch and returns only scalar metrics, not the coupling tensor. The existing `ms_cbtp_wjdot` computes per-source OT losses, but it stores only summary matrices, not reusable full `gamma_k` plans.

4. **Current plain WJDOT cannot obtain per-source target prediction `p_k`.**
   It has one shared classifier head and evaluates target predictions once. Existing posthoc CCSR uses source prototype probabilities as a proxy, not independently trained source expert predictions from WJDOT.

5. **Current code can compute source class prototypes, but plain WJDOT does not expose per-source prototypes as first-class artifacts.**
   `compute_class_prototypes()` exists in `src/tep_ot/ot_losses.py`. `ms_cbtp_wjdot` builds per-source prototypes internally for reliability weighting; posthoc CCSR recomputes source prototypes from collected embeddings. Plain `wjdot` does not save `mu_s[k,c]`.

6. **Current CCSR fusion can report raw-vs-final prediction differences, but the final prediction is intentionally conservative.**
   `export_ccsr_wjdot_fusion_artifacts()` computes `p_wjdot`, prototype-proxy `p_ccsr`, and `p_final`. It reports `raw_disagreement_count` and `final_disagreement_count`; with the current `safe_mix`/rho path, `final_confusion_matrix` can be very close to WJDOT even when raw CCSR differs.

7. **Current gate can be too conservative.**
   The active path is a safe mixture controlled by `lambda_safe`, `rho_min/rho_max`, entropy, agreement, and optional source-agreement override. It does not yet implement the requested `ccsr_raw`, `ccsr_safe`, and `ccsr_calibrated_override` modes with source-only meta calibration.

8. **CoDATS hidden dim is not matched to WJDOT/source-only by default.**
   `configs/method/codats.yaml` sets `backbone.classifier_hidden_dim: 500`, while `wjdot.yaml`, `ccsr_wjdot_fusion.yaml`, `source_only.yaml`, and `target_only.yaml` use 128 by default. This makes the current CoDATS comparison an official-capacity comparison, not a capacity-matched comparison.

9. **`target_only` currently uses labeled target train samples and should be named `target_ref` / `target_supervised_reference` in UDA result tables.**
   `train_benchmark.py` sets `protocol_override.use_target_labels = True` when `method_name == "target_only"`. The target train loader then uses true target train labels. It does not use target eval fold labels for training, but it is still a supervised target-domain reference rather than a UDA upper bound unless backbone, hidden dim, data amount, and train steps are explicitly matched.

## Checkpoint Selection And Leakage

- Checkpoint selection is trainer-controlled through `runtime.model_selection` and `runtime.early_stopping_metric`.
- The current multi-source config uses target-label-free proxy metrics for WJDOT/CCSR (`hybrid_source_eval_target_confidence`) and disables target eval during training.
- Target eval labels are used for final metrics, confusion matrices, and post-training CCSR reports only.

## Plotting And Tables

- Confusion matrix tables are exported by `_export_target_metric_tables()` in `src/trainers/train_benchmark.py`.
- Review/figure exports are driven by `export_analysis_artifacts()` and `src/evaluation/report_figures.py`.
- CCSR-specific heatmaps and disagreement tables are generated in `src/evaluation/ccsr_wjdot_fusion.py`.

## Implementation Implications

- Keep current pooled WJDOT as `pooled_wjdot`.
- Add source-aware WJDOT methods that keep `source_batches` separate and compute one OT loss per source.
- Add multi-head source experts so `p_k` is an actual expert prediction rather than only a prototype proxy.
- Expose per-source/class OT summaries and source/class alpha matrices through `reliability_snapshot()` and CCSR export tables.
- Add `target_ref` as the preferred supervised target-domain reference name while leaving `target_only` as a backward-compatible alias.
- Add capacity-matched configs for CoDATS/WJDOT/SA-CCSR at hidden dim 128 and 500.
