# Multi-Source WJDOT Code Audit

## Scope

This audit was performed inside the current workspace only. The current benchmark stack already has multi-source data loading and batch-plan expansion, while the WJDOT family is implemented under `src/methods/wjdot.py`.

## Existing Entry Points

- Data loader: `src/datasets/te_torch_dataset.py`
  - `prepare_benchmark_data()` materializes one or more source domains and one target domain.
  - Multi-source runs expose `source_train_loaders`, `source_train_eval_loaders`, and `source_eval_loaders` as lists, one loader per source domain.
- Fold split: `src/datasets/te_da_dataset.py`, `src/utils/fold_policy.py`, and `src/automation/run_small_scale_round.py`
  - Raw pickle `Folds` are authoritative.
  - Automation supports per-scene `source_folds_by_domain` and `target_fold`.
- Methods:
  - `source_only`: `src/methods/source_only.py`
  - `codats`: `src/methods/codats.py`
  - `wjdot` / WJDOT family: `src/methods/wjdot.py`
  - `target_only`: `src/methods/target_only.py`
- Method registry: `src/methods/__init__.py`
- Training and checkpoint selection: `src/trainers/train_benchmark.py`
- Result logger: `src/trainers/train_benchmark.py`, `src/evaluation/evaluate.py`, `src/evaluation/review.py`
- Per-run analysis artifacts: `analysis.npz` from `export_analysis_artifacts()`
- Confusion matrix / per-class recall tables: `_export_target_metric_tables()` in `src/trainers/train_benchmark.py`
- t-SNE / heatmap / confusion figures: `src/evaluation/report_figures.py`
- Multi-source config/batch support: `src/automation/run_small_scale_round.py`

## 1. How Current WJDOT Handles Multi-Source

There are two relevant code paths:

1. `wjdot` (`WJDOTMethod`) has `supports_multi_source = True`, but its `compute_loss()` calls `merge_source_batches()`. In a multi-source setting, source batches are concatenated into one supervised source batch, and one shared encoder/classifier is trained against the target batch with a single JDOT/WJDOT transport loss.

2. `ms_cbtp_wjdot` (`MSCBTPWJDOTMethod`) is a more advanced multi-source WJDOT-family implementation. It keeps each source batch separate inside `compute_loss()`, computes per-source OT losses, and aggregates them with source/class reliability weights.

Therefore, the method named `wjdot` currently performs multi-source training by source pooling, while `ms_cbtp_wjdot` contains the richer multi-source weighting machinery.

## 2. Does Current WJDOT Learn Global Source Alpha_k?

- `wjdot`: No. It merges source batches and does not learn or export a global source-domain weight `alpha_k`.
- `ms_cbtp_wjdot`: Yes, in the `global_alpha` mode. `_source_alpha()` computes source weights from source-target feature distances and uses them to aggregate per-source OT losses. The values are exposed through `alpha_source_*` metrics and `reliability_snapshot()`.

## 3. Does Current Code Have Per-Source OT Plan / Per-Source Prediction?

- Per-source OT diagnostics:
  - `ms_cbtp_wjdot` computes a per-source transport loss and records per-source/per-class OT mass and cost diagnostics.
  - `jdot_transport_loss()` currently returns the loss and scalar diagnostics, not the full transport plan `gamma`.
- Full per-source OT plan:
  - Not currently exported.
- Per-source prediction:
  - Not currently available for `wjdot`, because the model has one shared classifier.
  - `ms_cbtp_wjdot` also uses one shared classifier; it separates OT losses by source but does not export one target probability vector per source.

## 4. Can We Get Each Source's Target Prediction Probability?

Not directly from the current WJDOT classifier. The minimal stable option is to construct source-specific prediction proxies after training:

- collect source embeddings per source domain;
- compute normalized source class prototypes `mu_s[k,c]`;
- compute target embeddings;
- produce source-specific prototype probabilities `p_k[j,c] = softmax(-d(z_t[j], mu_s[k,c]) / T)`.

This is target-label-free and fits the requested first version when no source-specific WJDOT head exists.

## 5. Can We Get Source Class Prototypes and Target Embeddings?

Yes.

- `export_analysis_artifacts()` already collects source/target embeddings and logits.
- During final evaluation, the trainer also has direct access to `prepared_data.source_train_eval_loaders`, `prepared_data.source_eval_loaders`, and `prepared_data.target_train_loader`.
- `src/tep_ot/ot_losses.py` already provides `compute_class_prototypes()`.

The only missing piece is a dedicated final-stage fusion/export routine that computes prototypes, reliability components, alpha matrices, final predictions, and paper-friendly diagnostics without using target labels for reliability.

## 6. Minimal Changes for CCSR-WJDOT

The lowest-risk first version is prediction-level fusion:

1. Add `ccsr_wjdot_fusion` as a WJDOT-family method alias/class so it trains like current `wjdot`.
2. Add a final evaluation hook for `ccsr_wjdot_fusion`.
3. In that hook:
   - use target-train unlabeled data to estimate reliability;
   - use source labels only for source prototypes and source validation recall;
   - use target-eval labels only after predictions are fixed, for final metrics;
   - compute `alpha_{k,c}` with prototype distance, OT/probability proxy cost, prediction entropy, and source validation error;
   - compute `p_ccsr`, safety `rho_j`, and final fallback prediction `p_final`.
4. Save:
   - `reliability_components.csv`
   - `class_source_alpha.csv`
   - `class_source_alpha_heatmap.png`
   - `reliability_component_heatmaps.png`
   - `per_class_recall_gain.csv`
   - `ccsr_vs_wjdot_prediction_disagreement.csv`
   - `source_weight_global_vs_class_conditional.csv`
   - target prediction histogram and entropy/rho diagnostics.
5. Add configs for:
   - `wjdot`
   - `ccsr_wjdot_fusion`
   - the requested 3 two-source and 6 five-source fold0 tasks.

This avoids rewriting the training loop or WJDOT loss, keeps checkpoint selection unchanged, and makes CCSR a safe enhancement over the original WJDOT prediction.
