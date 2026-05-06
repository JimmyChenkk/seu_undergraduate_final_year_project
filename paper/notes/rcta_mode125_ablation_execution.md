# RCTA mode1/2/5 消融实验执行备忘录

## 实验口径

当前阶段将论文实验范围收束到 mode1、mode2、mode5 三个域：

- 主 benchmark：8 种方法 x 9 个 mode1/2/5 握手场景 = 72 run。
- RCTA 递进消融：5 个 RCTA 变体 x 同样 9 个场景 = 45 run。
- 所有场景的 fold 分配均沿用 `configs/experiment/benchmark_88_2clusters_11scenes_8methods_fixedfold.yaml` 中当前 mode1/2/5 设置。

这样做的目的不是强行追求每个场景都完全符合 `goal.md` 的理想排序，而是构造一个可写论文的证据链：

1. 主 benchmark 证明完整 RCTA 在统一 fixed-fold 设置下总体有效。
2. 递进消融说明 RCTA 的收益来自多个机制的互补叠加。
3. 机制指标解释为什么某些场景增益明显、某些场景不单调。

## 消融变体

| 变体 | 配置 | 论文含义 |
| --- | --- | --- |
| M0 | `rcta_m0_base_da` | hybrid DA + MCC 基础对齐底座 |
| M1 | `rcta_m1_temporal_mt` | M0 + EMA teacher + 弱强时序一致性 |
| M2 | `rcta_m2_reliability_gate` | M1 + 可靠性门控伪标签 |
| M3 | `rcta_m3_dual_proto_static` | M2 + 双原型记忆和类条件结构约束 |
| M4 | `rcta_m4_full` | 完整 RCTA：课程调度、置信保护选择、多源可靠性加权 |

论文中可以把 M0 和 M1 一起服务于第三章：M0 是对齐底座，M1 才是教师--学生时序一致性模块。这样就不会让章节结构被 5 个变体切得太碎。

## 运行命令

只检查计划，不启动训练：

```bash
scripts/run_rcta_mode125_ablation.sh --plan-only
```

正式运行 45 个消融实验：

```bash
scripts/run_rcta_mode125_ablation.sh
```

指定一个更容易识别的 batch 名称：

```bash
scripts/run_rcta_mode125_ablation.sh \
  --batch-root-name 202604xx_rcta_mode125_ablation_fixedfold
```

实验结束后汇总递进消融结果：

```bash
conda run -n tep_env python scripts/summarize_rcta_ablation.py \
  runs/202604xx_rcta_mode125_ablation_fixedfold
```

输出文件会写入：

- `comparison_summary/tables/rcta_ablation_summary.md`
- `comparison_summary/tables/rcta_ablation_summary.csv`
- `comparison_summary/tables/rcta_ablation_rows.csv`
- `comparison_summary/tables/rcta_ablation_summary.json`

## 论文回填顺序

1. 先回填第七章主 benchmark 的单源表、多源表、总体汇总表。
2. 再回填 `RCTA 模块递进消融总体结果` 表。
3. 最后回填 `RCTA 消融实验机制指标` 表。

机制指标优先使用：

- `gate_accept_ratio`
- `pseudo_kept_mean_reliability`
- `pseudo_label_kept_classes`
- `source_weight_entropy`

其中 `source_weight_entropy` 主要用于多源场景解释。单源场景中该值通常没有实际区分意义。

## 结果解释原则

不要写：

> 每个模块在所有迁移场景中均带来线性提升。

建议写：

> 消融结果表明，随着教师--学生一致性、可靠性门控、双原型约束和课程化训练逐步引入，模型在平均性能和伪标签可靠性等指标上呈现递进改善。个别场景中模块增益不完全单调，说明无监督跨工况迁移仍然受到域间偏移方向、类别混淆和伪标签质量的共同影响。

如果 M3 到 M4 的准确率提升不大，也可以从机制指标解释：

- gate 接纳比例是否更合理；
- 伪标签平均可靠性是否更高；
- 多源权重是否不再平均分配；
- 混淆矩阵是否减少了少数类别塌缩。

如果 M2 或 M3 某些场景下降，不要急着改算法。优先把它写成“模块交互具有场景依赖性，完整 RCTA 通过课程调度和保护型选择缓解该问题”。
