# CA-CCSR-WJDOT Code Audit

This audit was performed inside the current workspace only.

## Files Read

- `src/methods/codats.py`
- `src/methods/base.py`
- `src/backbones/fcn.py`
- `src/losses/domain.py`
- `src/methods/wjdot.py`
- `src/methods/__init__.py`
- `src/tep_ot/ot_losses.py`
- `src/evaluation/ccsr_wjdot_fusion.py`
- `src/evaluation/ccsr_wjdot_posthoc.py`
- `src/trainers/train_benchmark.py`
- `src/trainers/selection_metrics.py`
- `src/datasets/te_torch_dataset.py`
- `src/evaluation/evaluate.py`
- `src/evaluation/report_figures.py`
- `src/automation/run_small_scale_round.py`
- `configs/data/te_da.yaml`
- `configs/method/codats.yaml`
- `configs/method/codats_128.yaml`
- `configs/method/codats_500.yaml`
- `configs/method/wjdot.yaml`
- `configs/method/sourceaware_wjdot_shared_head.yaml`
- `configs/method/sourceaware_wjdot_multi_head.yaml`
- `configs/method/sa_ccsr_wjdot_train_128.yaml`
- `configs/method/ccsr_wjdot_fusion.yaml`
- `configs/method/target_ref.yaml`
- `configs/method/target_only.yaml`
- `configs/experiment/tep_ot_multisource_sa_ccsr_wjdot_stage1_probe_fold0.yaml`
- `configs/experiment/tep_ot_multisource_ccsr_wjdot_stage1_fold0.yaml`
- `configs/experiment/tep_ot_multisource_capacity_fairness_probe_fold0.yaml`
- `tests/test_wjdot_methods.py`
- `tests/test_ccsr_wjdot_fusion.py`
- `tests/test_no_target_label_leakage.py`
- `tests/test_automation_plan.py`

## Executive Summary

The current workspace already has most of the raw ingredients for CA-CCSR-WJDOT:

- CoDATS temporal encoder/classifier/discriminator exists and is trainable in the same benchmark interface.
- Source-aware WJDOT already computes separate source-target OT evidence per source.
- SA-CCSR-WJDOT training already computes a class-source alpha matrix and applies it to class-wise OT loss.
- Posthoc CCSR fusion already writes reliability components, alpha heatmaps, source meta-calibration tables, and disagreement diagnostics.

The missing piece is the actual new main method:

`ca_ccsr_wjdot = CoDATS backbone/teacher + per-source WJDOT in CoDATS feature space + CCSR class-source reliability + teacher-safe distillation/fusion`.

Current CCSR/WJDOT code is still WJDOT-centered. It does not yet initialize from a CoDATS checkpoint, keep a frozen CoDATS teacher, train with CoDATS adversarial loss during WJDOT adaptation, or compare/fuse against CoDATS teacher predictions at inference.

## 1. Current CoDATS Encoder / Classifier / Domain Discriminator

Location:

- CoDATS method: `src/methods/codats.py`
- Shared encoder/classifier base: `src/methods/base.py`
- FCN temporal encoder: `src/backbones/fcn.py`
- Domain discriminator and GRL: `src/losses/domain.py`
- Registry: `src/methods/__init__.py`
- Configs: `configs/method/codats.yaml`, `configs/method/codats_128.yaml`, `configs/method/codats_500.yaml`

Details:

- `CoDATSMethod` subclasses `SingleSourceMethodBase`.
- The encoder is built by `SingleSourceMethodBase.__init__()` through `build_backbone()`.
- With the current configs, the backbone is `fcn`, implemented by `FullyConvolutionalEncoder`.
- `FullyConvolutionalEncoder` is a 1D FCN over TEP signals shaped `(N, C, T)`:
  - Conv1d 34 -> 128, kernel 9
  - Conv1d 128 -> 256, kernel 5
  - Conv1d 256 -> 128, kernel 3
  - temporal global average pooling
  - output feature dimension `128`
- CoDATS replaces the default `ClassifierHead` with `CoDATSClassifierHead`, a two-hidden-layer MLP:
  - Linear feature_dim -> hidden_dim
  - ReLU / dropout
  - Linear hidden_dim -> hidden_dim
  - ReLU / dropout
  - Linear hidden_dim -> num_classes
- The domain discriminator is `DomainDiscriminator` from `src/losses/domain.py`.
- Adversarial alignment uses `domain_adversarial_loss()` with `WarmStartGradientReverseLayer` or `GradientReverseLayer`.

Important config point:

- `configs/method/codats.yaml` uses classifier hidden dim `500`.
- `configs/method/codats_128.yaml` uses hidden dim `128`.
- `configs/method/codats_500.yaml` uses hidden dim `500`.
- CA-CCSR-WJDOT must match whichever CoDATS baseline is used in the main table.

## 2. Does Current CoDATS Support Multi-Source Training?

Partially yes, but only as pooled-source multi-source training.

`SingleSourceMethodBase.supports_multi_source = True`, and `CoDATSMethod.compute_loss()` calls:

- `self.merge_source_batches(source_batches)`

For multiple source domains, `merge_source_batches()` concatenates all source minibatches into one `source_x` and one `source_y`. CoDATS then trains:

- one source CE loss on pooled sources;
- one binary source-vs-target adversarial discriminator;
- no source-domain-specific discriminator labels;
- no per-source reliability or per-source teacher evidence.

Conclusion:

- Current CoDATS can run in multi-source experiments without crashing.
- It does not preserve source identity during CoDATS adversarial training.
- For CA-CCSR-WJDOT this is acceptable for the teacher/backbone stage if the baseline CoDATS is also pooled-source, but the WJDOT/CCSR stage must keep source batches separate.

## 3. How To Get CoDATS Feature Embedding And Target Prediction

Existing API:

- `model(x)` returns `(logits, features)` through `SingleSourceMethodBase.forward()`.
- `model.predict_logits(x)` returns logits.
- `model.extract_features(x)` returns features.

Existing collection paths:

- `export_analysis_artifacts()` in `src/trainers/train_benchmark.py` collects:
  - source embeddings
  - source labels
  - source predictions
  - target embeddings
  - target labels, only for final evaluation artifacts
  - target predictions
  - target logits
- `_collect_loader_outputs()` in `src/trainers/train_benchmark.py` also collects logits, predictions, labels, and embeddings.
- `_collect_outputs()` in `src/evaluation/ccsr_wjdot_fusion.py` does the same and additionally calls `model.source_expert_probabilities(x)` when the model exposes per-source expert probabilities.

Reusable for CA:

- CoDATS teacher target probabilities can be obtained with `softmax(teacher.predict_logits(x_t))`.
- CoDATS teacher target embeddings can be obtained with `teacher(x_t)[1]` or `teacher.extract_features(x_t)`.
- The existing analysis export can persist teacher/student embeddings if extended with teacher-specific keys.

Missing:

- There is no current helper that simultaneously collects frozen teacher outputs and student outputs for the same batches.
- There is no `codats_teacher_metrics.csv` export yet.

## 4. Reusability Of WJDOT Per-Source OT Plan / Prediction / Class-Wise Transport Loss

Reusable pieces:

- `src/methods/wjdot.py::_sourceaware_transport_loss()` computes one source-specific WJDOT coupling:
  - feature cost: `cdist(source_features, target_features)^2`
  - label cost: source-label CE against target logits
  - normalized joint plan cost
  - `gamma = solve_coupling(...)`
  - total OT loss
  - class-wise OT cost vector
  - class-wise transported mass vector
  - detached OT plan `gamma_detached`
- `SourceAwareWJDOTSharedHeadMethod._compute_sourceaware_terms()` loops over source batches and keeps:
  - `source_ot_losses`
  - `source_class_losses`
  - `source_class_mass`
  - `gamma_plans`
  - `source_target_probs`
  - per-source prediction histogram
- `SACCsrWJDOTTrainMethod` already aggregates class-wise source transport loss with `class_alpha`.
- `reliability_snapshot()` exposes source/class alpha and matrix diagnostics for table export.

Limitations:

- Plain `WJDOTMethod` pools source batches and only computes one pooled OT loss.
- `jdot_transport_loss()` in `src/tep_ot/ot_losses.py` returns only loss and scalar metrics, not the full gamma.
- `SourceAwareWJDOTSharedHeadMethod` stores `_last_source_ot_plans`, but `reliability_snapshot()` does not export full OT plans. This is fine for metrics, but not enough if later diagnostics need full plan files.
- Shared-head per-source prediction is not a true per-source expert. In `SourceAwareWJDOTSharedHeadMethod`, `source_expert_probabilities()` repeats the same shared classifier probabilities for each source. True source-specific probabilities only exist in `SourceAwareWJDOTMultiHeadMethod`.

Recommendation for CA:

- Use the source-aware shared-head path as the base because the user explicitly wants to avoid complex multi-head by default.
- Reuse `_sourceaware_transport_loss()` and `_compute_sourceaware_terms()`, but swap in CoDATS classifier/head and add CoDATS adversarial loss.
- If per-source prediction support is needed for fusion, start with prototype/OT evidence or optional calibration heads, not default multi-head.

## 5. Reusability Of CCSR Alpha / Reliability Components / Heatmap

Training-time CCSR:

- `SACCsrWJDOTTrainMethod` computes:
  - `D_proto`
  - `D_ot`
  - `H_pred`
  - `E_src`
  - normalized components
  - reliability matrix `R`
  - class-source `alpha`
- It applies `alpha[k,c]` to class-wise OT loss in `_aggregate_class_transport_loss()`.
- It saves snapshot entries through `reliability_snapshot()`.

Post-training CCSR:

- `export_ccsr_wjdot_fusion_artifacts()` writes:
  - `reliability_components.csv`
  - `class_source_alpha.csv`
  - `class_source_alpha_matrix.csv`
  - `source_weight_global_vs_class_conditional.csv`
  - `global_source_alpha_vs_class_source_alpha.csv`
  - `per_class_recall_gain.csv`
  - `ccsr_vs_wjdot_prediction_disagreement.csv`
  - `target_prediction_histogram.csv`
  - `per_source_prediction.csv`
  - `per_source_confusion_on_target_eval_only_after_training.csv`
  - `target_prediction_histogram_per_source.csv`
  - `per_source_ot_loss.csv`
  - `per_source_alpha.csv`
  - `source_meta_calibration_results.csv`
  - `selected_gate_params.json`
  - `class_source_alpha_heatmap.png`
  - `reliability_component_heatmaps.png`
  - `global_source_alpha_vs_class_alpha.png`
  - `target_entropy_rho_distribution.png`
  - `wjdot_vs_ccsr_confusion_matrix.png`

What needs adjustment for CA:

- Reliability must be computed in CoDATS feature space.
- Training-time `E_src` in `SACCsrWJDOTTrainMethod` is currently computed from the current source minibatch logits, not from full source validation recalls. Posthoc CCSR uses source eval loaders and is closer to the requested definition.
- Current target prototype fallback in training uses source target probabilities, not OT-induced target barycenters from `gamma_k`.
- Posthoc CCSR uses prototype-proxy target probabilities when source experts are unavailable. CA should prefer teacher/student probabilities and optionally OT class evidence `q_ot`.
- Current component weights default to roughly `0.35/0.35/0.20/0.10` in training. The new CA spec asks for `0.30/0.35/0.20/0.15`.
- Current heatmap functions are reusable, but filenames for CA should include the requested aliases where different:
  - `global_source_alpha_vs_class_alpha.csv`
  - `alpha_entropy_per_class.csv`
  - teacher/student disagreement files

## 6. Can Checkpoint Selection Record Both CoDATS Teacher Score And CA-CCSR-WJDOT Score?

Not yet.

Current trainer behavior:

- `run_deep_experiment()` trains exactly one model instance.
- `history` records one model's epoch metrics.
- checkpoint selection stores one selected state dict.
- `_export_checkpoint_diagnostics()` writes one set of target-free proxy diagnostics and one `model_selection_score`.

Existing useful parts:

- `save_checkpoint` writes `model.pt`.
- `result.json` records `checkpoint_path`.
- `ccsr_wjdot_posthoc.py` already knows how to load a previous run checkpoint.

Missing for CA:

- Stage A CoDATS teacher training inside the CA run, or an automation dependency that trains CoDATS first and passes its checkpoint to CA.
- Teacher metrics recorded in CA output:
  - `codats_teacher_metrics.csv`
  - teacher target train confidence/entropy
  - teacher source train/eval metrics
  - teacher final target metrics only after final evaluation
- Student-vs-teacher checkpoint diagnostics:
  - CA student selection score
  - teacher score carried forward as a reference
  - final fusion score only after predictions are fixed
- A safe way to initialize student from teacher checkpoint and freeze a teacher copy.

Conclusion:

The current checkpoint system can save/load individual checkpoints, but it cannot yet manage the requested two-model teacher/student CA checkpoint selection.

## 7. Are Target Labels Used Only In Final Evaluation?

For UDA training batches:

- Default data config has `use_target_labels: false`.
- `prepare_benchmark_data()` masks target-train labels to `-1` when `target_label_mode == "unlabeled"`.
- Existing WJDOT target-label leakage tests confirm unsupervised WJDOT variants ignore target batch labels unless `target_label_assist_weight` is explicitly enabled.

For target eval labels:

- The generic trainer default is `target_eval_during_training = true`.
- The current multi-source experiment configs set `target_eval_during_training: false`, which is correct for the requested main line.
- Even when `target_eval_during_training: false`, the trainer still evaluates the final selected model on `target_eval_loader` after training.
- Selection metrics registry still contains `target_eval` / `best_target_eval` / `best_target_eval_oracle`; these must not be used for UDA configs.

For CCSR:

- `export_ccsr_wjdot_fusion_artifacts()` documents that reliability estimation uses source labels and unlabeled target-train samples.
- Target-eval labels are read only after final predictions are fixed, for final metrics and diagnostic tables.
- Some diagnostic filenames explicitly say `on_target_eval_only_after_training`, which is acceptable for final analysis but not for calibration or selection.

For target reference:

- `train_benchmark.py` forces `use_target_labels = true` for `target_only`, `target_ref`, and related aliases.
- Therefore `target_ref` is a supervised target-domain reference, not a UDA baseline.

Conclusion:

- In the current multi-source configs, target eval labels are not used for training/selection because `target_eval_during_training: false` and target-free selection metrics are configured.
- In the generic trainer, target eval labels are not guaranteed to be final-only because the default is `true` and target-eval selection metrics exist.
- CA configs should explicitly set:
  - `target_eval_during_training: false`
  - target-free `model_selection`
  - target-free `early_stopping_metric`
  - `target_label_assist_weight: 0.0`

## 8. Files To Modify For `ca_ccsr_wjdot`

Core method implementation:

- `src/methods/wjdot.py`
  - Add `CACcsrWJDOTMethod`.
  - Prefer subclassing the source-aware shared-head path, not multi-head.
  - Replace/initialize classifier with `CoDATSClassifierHead`.
  - Add CoDATS adversarial loss and discriminator.
  - Add frozen teacher modules or teacher checkpoint loading.
  - Add teacher anchor distillation loss.
  - Add CA-specific reliability components and class-wise OT aggregation.
  - Add teacher-safe prediction/fusion helpers or output hooks.

- `src/methods/codats.py`
  - Reuse `CoDATSClassifierHead`.
  - Potentially add small helper functions for building CoDATS discriminator/GRL if duplicating code becomes too noisy.

- `src/methods/__init__.py`
  - Register `ca_ccsr_wjdot`.
  - Parse CA loss config:
    - CoDATS adv params
    - `lambda_ot`
    - `lambda_ccsr`
    - `lambda_teacher`
    - reliability schedule
    - alpha top-m/floor/temperature
    - teacher-safe fusion thresholds

Training and checkpoint orchestration:

- `src/trainers/train_benchmark.py`
  - Add `ca_ccsr_wjdot` to OT dependency set.
  - Add teacher checkpoint initialization path or internal Stage A/Stage B training flow.
  - Save `codats_teacher_metrics.csv`.
  - Save CA loss curves:
    - `wjdot_loss_curve.csv`
    - `ccsr_loss_curve.csv`
    - `teacher_anchor_loss_curve.csv`
  - Extend checkpoint diagnostics with teacher/student fields.
  - Add CA final artifact hook.
  - Preserve target-label-free model selection.

- `src/automation/run_small_scale_round.py`
  - Ensure Stage 1 method order includes CoDATS before CA if CA depends on a trained teacher checkpoint.
  - Add a non-posthoc dependency mapping or train CA internally with its own Stage A teacher.
  - Keep posthoc CCSR methods out of the final main method list.

Evaluation and CA diagnostics:

- Add a new module such as `src/evaluation/ca_ccsr_wjdot.py`, or carefully extend `src/evaluation/ccsr_wjdot_fusion.py`.
  - Prefer a new CA-specific module to avoid mixing posthoc WJDOT fusion with the final CA method.
  - Save:
    - `teacher_student_disagreement.csv`
    - `teacher_safe_fusion_summary.csv`
    - `eta_distribution.csv`
    - `override_cases.csv`
    - `per_class_recall_gain_vs_codats.csv`
    - `per_class_recall_gain_vs_wjdot.csv`
    - `alpha_entropy_per_class.csv`
    - CA-vs-CoDATS and CA-vs-WJDOT confusion comparisons.

Configs:

- Add `configs/method/ca_ccsr_wjdot.yaml`
  - `method_name: ca_ccsr_wjdot`
  - `method_display_name: CA-CCSR-WJDOT`
  - `classifier_hidden_dim` must match the chosen CoDATS baseline.
  - Default shared classifier, no multi-head.
  - `target_label_assist_weight: 0.0`

- Add final Stage 1 probe config, for example:
  - `configs/experiment/tep_ot_multisource_ca_ccsr_wjdot_stage1_probe_fold0.yaml`
  - methods:
    - `source_only`
    - `codats_128` or final chosen CoDATS display config
    - `wjdot`
    - `ca_ccsr_wjdot`
    - `target_ref`
  - tasks:
    - `mode1+mode5->mode2`
    - `mode1+mode2+mode4+mode5+mode6->mode3`
    - `mode1+mode2+mode3+mode4+mode6->mode5`
  - `target_eval_during_training: false`

Reporting:

- `src/evaluation/report_figures.py`
  - Add `ca_ccsr_wjdot` / `CA-CCSR-WJDOT` to method order and display map.
  - Hide intermediate methods from final main figures where needed.

- `src/evaluation/evaluate.py`
  - Add final main table export that only includes:
    - `source_only`
    - `CoDATS`
    - `WJDOT`
    - `CA-CCSR-WJDOT`
    - `target_ref`
  - Keep intermediate methods only for ablation/diagnostics.

- `scripts/summarize_benchmark.py`
  - Ensure final paper summary uses display names and final-method filtering.

Tests:

- `tests/test_wjdot_methods.py`
  - Add method registry/compute-loss tests for `ca_ccsr_wjdot`.
  - Add target-label invariance test for CA when target labels differ.
  - Add reliability snapshot shape checks.

- `tests/test_ccsr_wjdot_fusion.py` or a new `tests/test_ca_ccsr_wjdot.py`
  - Add teacher-safe fusion diagnostic export tests.
  - Add alpha entropy and disagreement file tests.

- `tests/test_automation_plan.py`
  - Ensure final Stage 1 method list is the requested 5 methods.
  - Ensure CoDATS teacher dependency is ordered or handled.

- `tests/test_report_figures.py`
  - Ensure final method ordering and display names match:
    - `source_only`
    - `CoDATS`
    - `WJDOT`
    - `CA-CCSR-WJDOT`
    - `target_ref`

## Implementation Risk Notes

1. The fastest low-risk implementation is to make CA train its own teacher in the same run only if the trainer can support a two-stage method cleanly. Otherwise, use automation to run CoDATS first and pass its checkpoint into CA.

2. If CA is implemented as a normal single model without a frozen teacher, it will not satisfy the requested teacher anchor design.

3. If CA uses the existing posthoc CCSR export unchanged, it will still be WJDOT-centered and will not satisfy the new main-method definition.

4. If CA defaults to multi-head, it risks repeating the previously observed weak source-domain fitting. The default should be shared classifier.

5. The current generic trainer default `target_eval_during_training: true` is a footgun. CA experiment configs should explicitly disable it and use target-free selection metrics.

## Recommended Next Step

Implement `ca_ccsr_wjdot` as a CoDATS-initialized source-aware WJDOT/CCSR method with a frozen teacher copy, then add the Stage 1 probe config with only the final five methods.
