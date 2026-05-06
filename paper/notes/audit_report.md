# 项目代码与数据审计报告

生成时间：2026-05-01

本报告只基于当前仓库内代码、配置、manifest 和已生成的轻量级检查结果，不包含伪造实验结果。详细数据契约检查见 `paper/resources/dataset_integrity.md`，mode 距离代理结果见 `paper/resources/domain_distance_proxy_1245.json`。

## 1. 数据读取与 TEP benchmark 契约

- 当前 manifest 包含 6 个 domain：mode1 到 mode6。
- 每个 domain 的 `Signals` 形状均为 `(N, 600, 34)`，训练前在 `channels_first=true` 下转置为 `(N, 34, 600)`。
- 34 个通道与 TEP DA benchmark 常用的连续变量约定一致，可解释为 XME(1)-XME(22) 与 XMV(1)-XMV(12)。当前 raw pickle 不保存变量名，因此只能从通道数与数据来源契约确认，不能从文件内字段名二次验证。
- 每个 domain 的标签范围为 0 到 28，唯一标签数为 29，符合 normal/no fault + 28 faults 的分类设定。
- 当前没有证据表明读取了额外非连续变量或显式泄漏变量；`slice_domain_split()` 只读取 `Signals` 和 `Labels`。

## 2. 标准化审计

- `configs/data/te_da.yaml` 当前使用 `normalization=standardization`、`normalization_scope=domain`。
- `src/datasets/te_torch_dataset.py` 在该设置下先对完整 domain 计算统计量，再切分 train/eval fold。这与本地 benchmark 注释一致，但严格 UDA 口径下会使用目标 eval fold 的无标签统计量。
- 结论：这是 benchmark-level / mode-level 标准化，不是严格 train-only 标准化。论文中必须明确说明；若答辩老师追问，应解释为无标签目标域整体可见的 benchmark 预处理，而不是使用目标标签。
- 代码已保留 `normalization_scope=train` 支持，可用于额外无泄漏敏感性实验。

## 3. Fold / Split 审计

- 当前 loader 策略是：选中 fold 作为 eval，其余样本作为 train。
- mode2 到 mode6 的 manifest 中 5 个 fold 长度之和略小于样本总数，例如 mode2 为 2835 vs 2845。这意味着有少量样本不在任何 eval fold 中，但在当前单次 fixed-fold 训练中仍会进入 train 集。
- fixed-fold / random-fold 的实际运行主要由 `src/automation/run_small_scale_round.py` 写入临时 experiment 配置；直接运行 `train_benchmark.py` 时不会自动消费 `scene_fold_overrides`。
- 已修正无监督训练安全性：当 `target_label_mode=unlabeled` 时，`target_train_loader` 返回 `-1` 哑标签；`target_only` 在训练入口被显式标记为使用目标标签，仅作为上界。

## 4. Source-Only / Target-Only 合理性检查

- 当前代码层面已区分：Source-Only 只使用源域标签；Target-Only 使用目标域标签，定位为监督上界。
- 尚未在本轮重新跑完整 source-only / target-only 数值，因此不能声称 target-only 一定高于 source-only。
- 若后续 target-only 不明显高于 source-only，应优先排查：标签映射、fold 设置、模型训练不足、目标域训练标签是否被错误屏蔽、以及 mode-level 标准化口径差异。

## 5. DeepJDOT 初步诊断

- 当前 `deepjdot_loss()` 使用 minibatch OT：特征平方距离 + 类别概率代价求 coupling，再用固定 coupling 优化特征距离与目标交叉熵。
- 已接通的关键参数包括 `reg_dist`、`reg_cl`、`normalize_feature_cost`、`transport_solver`、`sinkhorn_reg`、`sinkhorn_num_iter_max`。
- 高风险点：默认 `batch_size=32` 可能导致源/目标 batch 类别覆盖不足；`normalize_feature_cost=false` 时特征代价尺度可能压过类别代价；EMD 对异常 cost 更敏感；直接使用目标 logits 的即时概率可能导致早期 coupling 噪声。
- 已新增 `configs/experiment/rescue_deepjdot_stability.yaml`，用于 larger batch + warm-start + feature cost normalization + Sinkhorn 的稳定性版本。
- 结论暂定：不能简单说 DeepJDOT 不行；需要至少比较默认 EMD 版本和稳定化 Sinkhorn 版本后，再判断是实现问题、参数问题还是任务设定差异。

## 6. DANN / CDAN 表现差的可能原因

- DANN 只对齐边缘特征分布，TEP 29 类故障诊断中容易出现“域对齐但类别混叠”。
- CDAN 已作为新递进方法的基底，因为它把 feature 与 prediction conditioning 一起送入域判别器，更符合类条件对齐主线。
- 对抗分支的关键诊断指标包括 `domain_accuracy`、`lambda_domain`、`grl_coeff`、`mean_target_entropy`。若 domain accuracy 长期接近 1，说明对齐不足；若长期接近 0.5 但分类下降，说明可能过度域不变或类别混叠。

## 7. 125 与 145 场景的初步判断

- 已新增 `scripts/analyze_domain_distances.py`，并对 mode1/2/4/5 做了 256 样本轻量距离代理。
- 当前代理显示 1/4、1/5、4/5 的 combined proxy 明显小于涉及 mode2 的若干 pair；这支持“145 是边缘分布相对近的诊断簇”的说法。
- 但该代理基于 raw channel mean/cov，不等价于类条件故障分类难度。论文中应明确区分边缘分布相近与故障类别判别容易。
- 建议：主实验继续保留 125 九场景作为困难主线，同时在诊断实验中加入 1->4 或 4->5，用于解释 145 与 125 的差异。

## 8. 已完成修正

- 新增 TC-CDAN、RPL-TC-CDAN、CCS-RPL-TC-CDAN 三个方法文件和配置。
- 无监督目标域训练 loader 已屏蔽目标训练标签，Target-Only 入口显式启用目标标签。
- 新增数据完整性、三方法 smoke、无目标标签依赖和 prototype/pseudo 空样本安全测试。
- 新增论文图表占位导出脚本，所有占位图标注“示意图”或“待实验结果替换”。

## 9. 仍需人工确认 / 后续实验

- 需要实际运行 smoke test，确认三方法在 mode1->mode2 上 loss finite、指标完整。
- 需要跑 DeepJDOT 默认与稳定化版本，才能给出最终结论。
- 需要跑 125 与至少一个 145 诊断场景，确认边缘距离代理是否与分类指标一致。
- 若递进结果不单调，优先调小一致性/伪标签/原型权重，推迟 pseudo/prototype 启用，并降低 domain alignment 强度。
