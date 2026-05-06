# 本科毕业论文三项改良点实现原理梳理

本文档用于先把当前仓库中已经收敛到主线的三个改良点讲清楚，后续可直接发给网页 chatbot 一起讨论论文表述、章节命名和实验叙事。这里聚焦“代码如何实现”和“论文中可以怎样解释”，不是最终论文正文。

## 0. 当前实验主线

当前认可并需要收敛的实验主线分为单源 48 组与多源 45 组。

### 单源 48 实验

单源主线覆盖 `mode1`、`mode2`、`mode5` 三个域之间的 6 个有向迁移场景，每个场景运行 8 个方法，共 `8 x 6 = 48` 组。主线中的两个改良点是：

- `tpu_deepjdot`
- `cbtpu_deepjdot`

运行命令：

```bash
cd ~/workspace/bak_seu_undergraduate_final_year_project

bash scripts/run_small_scale_round.sh \
  --data-config configs/data/te_da.yaml \
  --experiment-config configs/experiment/tep_ot_single_source_8methods_stage1_fold0.yaml \
  --batch-root-name single_source_48_seed42_$(date +%Y%m%d_%H%M%S)
```

关键配置入口：

- `configs/experiment/tep_ot_single_source_8methods_stage1_fold0.yaml`
- `configs/method/tpu_deepjdot.yaml`
- `configs/method/cbtpu_deepjdot.yaml`
- `src/methods/deepjdot.py`

### 多源 45 实验

多源主线按当前 CA-CCSR-WJDOT 方向收敛，覆盖 9 个多源迁移场景，每个场景运行 5 个方法，共 `5 x 9 = 45` 组。主线中的改良点是：

- `ca_ccsr_wjdot_prior20`，其底层方法名为 `ca_ccsr_wjdot`

运行命令：

```bash
cd ~/workspace/bak_seu_undergraduate_final_year_project

bash scripts/run_small_scale_round.sh \
  --data-config configs/data/te_da.yaml \
  --experiment-config configs/experiment/tep_ot_multisource_ca_ccsr_wjdot_stage1_probe_fold0.yaml \
  --batch-root-name multisource_45_ca_seed42_$(date +%Y%m%d_%H%M%S)
```

关键配置入口：

- `configs/experiment/tep_ot_multisource_ca_ccsr_wjdot_stage1_probe_fold0.yaml`
- `configs/method/ca_ccsr_wjdot_prior20.yaml`
- `configs/method/ca_ccsr_wjdot.yaml`
- `src/methods/wjdot.py`
- `src/evaluation/ca_ccsr_wjdot.py`
- `src/automation/run_small_scale_round.py`

## 1. 总体研究问题

本仓库当前围绕 Tennessee Eastman Process 故障诊断中的无监督领域自适应展开。训练时可以使用源域有标签样本与目标域无标签样本，目标域测试标签只在训练结束后用于最终评价。

基础问题可以概括为：

给定源域样本 `(x_s, y_s)` 和目标域无标签样本 `x_t`，学习一个诊断模型，使其在目标工况或目标运行模式上仍能正确识别故障类别。

当前三项改良点的共同动机是：传统对齐方法容易出现三类问题。

1. 单源迁移中，DeepJDOT 的 minibatch OT 对齐可能受类别不平衡、早期表征不稳定和目标域动态形态差异影响。
2. 若直接使用目标伪标签，早期错误伪标签会放大偏差，尤其容易造成某些故障类别塌缩。
3. 多源迁移中，不同源域对不同故障类别的贡献并不相同，简单合并多源可能让不可靠源域干扰目标域学习。

因此当前主线形成了“单源先稳住 OT 对齐，再谨慎利用目标伪语义；多源再做类条件源域可靠性选择”的递进结构。

## 2. 创新点一：TPU-DeepJDOT

### 2.1 方法定位

`TPU-DeepJDOT` 可以理解为 Temporal-Prototypical Unbalanced DeepJDOT，即在单源 DeepJDOT 上加入：

- 非平衡最优传输；
- 源域类别原型；
- 时序统计形态代价；
- 源域监督对比 warmup。

它的定位不是引入目标伪标签，而是先让单源 OT 对齐更稳定、更符合流程工业时间序列数据结构。

### 2.2 基础 DeepJDOT 目标

原始 `deepjdot` 在每个 minibatch 中计算源域特征与目标域特征之间的传输计划 `gamma`。传输代价由两部分构成：

- 特征距离：源域特征与目标域特征的平方距离；
- 标签代价：源域真实类别与目标域预测分布之间的分类代价。

在代码中，这部分由 `src/methods/deepjdot.py` 的 `_transport_with_diagnostics()` 完成。基础总代价可理解为：

```text
C_base(i, j) = reg_dist * d(f_s_i, f_t_j) + reg_cl * CE(y_s_i, p_t_j)
```

其中 `f_s_i` 是源样本特征，`f_t_j` 是目标样本特征，`p_t_j` 是目标样本预测概率。

训练目标为：

```text
L = L_source_ce + lambda_align * L_transport
```

### 2.3 TPU 的改良机制

`tpu_deepjdot` 继承 `DeepJDOTMethod`，主要实现位于 `TPUDeepJDOTMethod`。

第一，使用非平衡 OT。配置中启用：

```yaml
transport_solver: sinkhorn_unbalanced
unbalanced_transport: true
uot_tau_s: 1.0
uot_tau_t: 1.0
```

这允许传输计划的行列边缘分布适度偏离均匀分布，从而缓解源域与目标域 minibatch 类别比例不一致的问题。代码中对应 `_solve_deepjdot_coupling()` 的 `ot.unbalanced.sinkhorn_unbalanced()` 分支。

第二，引入源域类别原型。模型为每个源域类别维护一个 EMA prototype：

```text
P_c <- momentum * P_c + (1 - momentum) * mean(f_s | y_s = c)
```

代码中 `_update_source_prototypes()` 使用 `prototype_momentum=0.95` 更新每类原型。之后 `_relative_prototype_cost()` 会计算目标样本相对不同源类原型的距离，并把相对原型代价加入 OT plan cost：

```text
C = C_base + lambda_proto * C_proto
```

这里的 `C_proto` 不是简单“离本类原型越近越好”，而是比较“源标签对应原型距离”与“其他可用类原型平均距离”的相对关系，并进行标准差归一化和上下界裁剪。这样更像一种类别结构约束。

第三，引入时序统计代价。`_temporal_descriptor()` 从原始时间窗口中提取：

- 各通道均值；
- 各通道标准差；
- 一阶差分均值；
- 一阶差分标准差。

然后 `_temporal_cost()` 计算源目标样本之间的时序描述子距离，并加入 OT 代价：

```text
C = C_base + lambda_proto * C_proto + lambda_temp * C_temp
```

这使得对齐不只看神经网络特征，也考虑流程信号窗口的动态形态。

第四，加入源域监督对比 warmup。`_supervised_contrastive_loss()` 只使用源域标签，让同类源域样本特征更接近、异类更分散。配置中 `supcon_warmup_only: true`，表示该约束主要用于训练早期稳定源域类别结构，之后让位给 OT 对齐。

### 2.4 训练目标概括

`TPU-DeepJDOT` 的训练目标可概括为：

```text
L_TPU =
  L_source_ce
  + lambda_supcon * L_source_supcon
  + lambda_align * L_UOT(C_base + lambda_proto*C_proto + lambda_temp*C_temp)
```

其中 `lambda_align`、`lambda_proto`、`lambda_temp` 都有 warmup / ramp 调度。

### 2.5 论文可写成的贡献点

这部分可以写成：针对 DeepJDOT 在工业时间序列域适应中对 minibatch 类别比例敏感、忽略故障类别结构和时序动态形态的问题，提出一种时序-原型约束的非平衡 DeepJDOT。该方法通过源域 EMA 类原型、时序统计描述子和非平衡传输共同修正传输代价，使源目标对齐更稳定、更符合流程数据的类别结构与动态特征。

## 3. 创新点二：CBTPU-DeepJDOT

### 3.1 方法定位

`CBTPU-DeepJDOT` 可以理解为 Confidence-Balanced TPU-DeepJDOT。它建立在 `TPU-DeepJDOT` 之上，进一步解决“目标域无标签样本如何被安全利用”的问题。

它不是直接把模型预测当伪标签，而是让三种无标签语义证据达成一致后，才接受目标样本参与伪监督训练。

三种证据分别是：

- `q_ot`：由 OT 传输计划推导出的目标样本类别分布；
- `q_cls`：由 EMA teacher 对目标弱增强样本给出的类别分布；
- `q_proto`：由目标特征到源域类别原型距离得到的类别分布。

### 3.2 三路伪语义融合

`q_ot` 来自 `_q_ot_from_gamma()`。它将传输计划 `gamma` 按源域类别聚合，得到每个目标样本接收了哪些源类质量：

```text
q_ot(j, c) = mass transported from source class c to target sample j
```

`q_cls` 来自 EMA teacher。`CBTPUDeepJDOTMethod` 内部复制一份 teacher encoder 与 teacher classifier，并在每次优化后用 `_ema_update_teacher()` 更新：

```text
theta_teacher <- decay * theta_teacher + (1 - decay) * theta_student
```

`q_proto` 来自 `_prototype_probabilities()`，即目标样本到各源类原型距离的 softmax。

三路分布通过 log-space 加权融合：

```text
log q_mix =
  q_ot_power    * log(q_ot)
  + q_cls_power * log(q_cls)
  + q_proto_power * log(q_proto)
```

然后重新归一化得到 `q_mix`。

### 3.3 可靠伪标签接收门控

代码中使用三类门控条件控制目标样本是否进入伪标签训练：

1. 三路 top-1 类别完全一致，或者三路分布的 JS disagreement 足够低；
2. `q_mix` 最大置信度高于随训练推进下降的阈值 `tau`；
3. `q_ot` 的归一化熵低于阈值，表示 OT 分配本身不太模糊。

配置中主线参数包括：

```yaml
tau_start: 0.94
tau_end: 0.84
q_ot_entropy_threshold: 0.78
js_threshold: 0.08
pseudo_max_acceptance: 0.70
```

`pseudo_max_acceptance` 会限制每个 batch 最多接收的目标样本比例，避免早期或某些场景中过度自信导致伪标签污染。

### 3.4 目标域训练项

被接收的目标样本会在强增强视图上计算 soft pseudo CE：

```text
L_pseudo = CE(student(strong_aug(x_t)), q_mix)
```

此外，模型还对中等置信样本施加 teacher consistency：

```text
L_consistency = KL(student(strong_aug(x_t)) || q_cls)
```

弱增强和强增强由 `_DeepJDOTTemporalAugmenter` 实现，包括 jitter、scaling、time mask 和 channel dropout。这样伪监督不只是记住原始输入，而是要求预测在合理扰动下稳定。

为了缓解目标伪标签类别塌缩，`_logit_adjusted()` 会根据 accepted pseudo labels 的类别频率对 logits 做调整。直观上，如果某类已经被接收过多，其对应 logit 会被相对抑制，使训练更接近类别均衡。

### 3.5 训练目标概括

`CBTPU-DeepJDOT` 的目标可概括为：

```text
L_CBTPU =
  L_source_ce
  + lambda_supcon * L_source_supcon
  + lambda_align * L_UOT
  + lambda_pseudo * L_pseudo
  + lambda_consistency * L_consistency
  + lambda_infomax * L_infomax
```

其中 `L_UOT` 仍沿用 TPU 的非平衡时序-原型 OT 代价；`L_pseudo` 和 `L_consistency` 都经过后期启动与 warmup，避免早期目标伪标签过早介入。

### 3.6 论文可写成的贡献点

这部分可以写成：在时序-原型非平衡 OT 对齐基础上，进一步提出三源一致性的置信均衡伪标签机制。该机制综合 OT 传输语义、EMA teacher 预测语义和源域原型语义，仅当三者一致或分歧足够低时接收目标样本，并通过接收比例上限、类别均衡 logit adjustment 与强弱增强一致性约束降低伪标签噪声。

与 TPU 相比，CBTPU 的核心价值是：从“只做可靠对齐”推进到“谨慎利用目标域无标签语义”，但仍然保持 target-label-free。

## 4. 创新点三：CA-CCSR-WJDOT / prior20

### 4.1 方法定位

多源主线方法为 `ca_ccsr_wjdot_prior20`，配置文件中的 `method_name` 是 `ca_ccsr_wjdot`，`method_display_name` 是 `ca_ccsr_wjdot_prior20`。

它可以理解为 CoDATS-Augmented Class-Conditional Source Reliability WJDOT。核心问题是：多源迁移时，不同源域对不同故障类别的可靠性不同，不能只做简单 pooled source，也不能只给每个源域一个全局权重。

因此该方法引入类条件源域可靠性矩阵：

```text
alpha[k, c] = source k 对 class c 的可靠权重
```

### 4.2 与普通 WJDOT 的区别

普通 `wjdot` 或 pooled-source 训练把多个源域合并，计算一个整体 OT 对齐。`SourceAwareWJDOTSharedHeadMethod` 则会逐源计算 WJDOT：

```text
source_1 -> target
source_2 -> target
...
source_K -> target
```

每个源域都会得到：

- 源域 CE loss；
- per-source OT loss；
- 每个类别的 OT cost；
- 每个类别的 transported mass；
- 该源域对目标样本的预测分布。

这些中间量由 `_compute_sourceaware_terms()` 收集。

### 4.3 CCSR 类条件可靠性

`CA-CCSR-WJDOT` 继承了 source-aware WJDOT 的逐源证据，并计算四类 reliability component。

1. `D_proto`：源域类别原型与目标域类别原型之间的距离。距离越小，说明该源域在该类别上与目标域结构更接近。
2. `D_ot`：该源域该类别的 OT transport cost。代价越低，说明该源域类别更容易对齐到目标域。
3. `H_pred`：该源域专家在目标域相关样本上的预测熵。熵越低，说明该源域在该类别上更确定。
4. `E_src`：源域自身类别 recall error。源域上该类都学不好时，不应强信任该源域。

这四项会按类别做 min-max normalization，然后线性组合：

```text
R[k, c] =
  w_proto * D_proto[k, c]
  + w_ot * D_ot[k, c]
  + w_entropy * H_pred[k, c]
  + w_source_error * E_src[k, c]
```

再通过 softmax 转成类条件源域权重：

```text
alpha[:, c] = softmax(-R[:, c] / T_class)
```

并带有 floor、top-m 与 ramp 机制，训练早期先使用均匀权重，后期逐步切换到 reliability-aware 权重。

主线配置中四个 component 权重为：

```yaml
w_proto: 0.30
w_ot: 0.35
w_entropy: 0.20
w_source_error: 0.15
```

### 4.4 CoDATS augmentation

`CA-CCSR-WJDOT` 不只是 WJDOT 加 CCSR，还接入了 CoDATS 风格的分类头与领域对抗分支。

在 `CACCSRWJDOTMethod.__init__()` 中：

- classifier 被替换为 `CoDATSClassifierHead`；
- 新增 `DomainDiscriminator`；
- 通过 `WarmStartGradientReverseLayer` 或 `GradientReverseLayer` 做源-目标领域对抗；
- 使用 `domain_adversarial_loss()` 训练域不可分特征。

因此训练目标中包含 CoDATS 风格的对抗对齐项：

```text
lambda_adv * L_domain_adv
```

这部分的作用是给 WJDOT/CCSR 一个更稳定的跨域特征空间，而不是完全依赖 OT 代价。

### 4.5 Teacher anchor 与自动化 checkpoint 注入

`CA-CCSR-WJDOT` 使用冻结的 CoDATS teacher 作为 anchor。自动化脚本中 `_teacher_base_method_for()` 将所有 `ca_ccsr_wjdot*` 方法映射到 `codats`。批量运行时，`src/automation/run_small_scale_round.py` 会要求同一 batch、同一场景里先完成 `codats`，然后把它的 checkpoint 注入到 CA 方法配置的：

```yaml
loss.teacher_checkpoint_path
```

训练时 `train_benchmark.py` 会调用 `load_teacher_checkpoint_state()` 加载 teacher。若 checkpoint 存在，则 teacher anchor 通过 KL 或 MSE 约束 student 不要偏离已稳定的 CoDATS 表征/预测过远：

```text
lambda_teacher * L_teacher_anchor
```

当前 `prior20` 主线强化了 teacher anchor，并把训练设为较短的 refinement：

```yaml
epochs: 20
learning_rate: 0.00003
lambda_adv: 0.15
lambda_ot: 0.04
lambda_ccsr: 0.04
lambda_teacher: 0.20
```

这说明论文中可以把它表述为“以 CoDATS 稳定模型为教师先验，在其基础上进行类条件多源 OT refinement”。

### 4.6 训练目标概括

`CA-CCSR-WJDOT` 的训练目标可概括为：

```text
L_CA =
  source_ce_weight * L_source_ce
  + lambda_adv * L_domain_adv
  + lambda_ot * ramp_ot * L_sourceaware_WJDOT
  + lambda_ccsr * ramp_ccsr * L_class_conditional_CCSR
  + lambda_teacher * L_teacher_anchor
```

其中：

```text
L_class_conditional_CCSR =
  sum_{k,c} alpha[k,c] * OT_cost[k,c]
```

也就是说，最终不是让每个源域对所有类别贡献相同，而是让更可靠的源域在对应类别上贡献更大。

### 4.7 Final teacher-safe fusion

训练结束后，`src/evaluation/ca_ccsr_wjdot.py` 会导出最终诊断与解释性诊断表。核心步骤是：

1. 收集 student 在目标评估集上的 logits；
2. 收集 teacher 在同一批样本上的 logits；
3. 在读取目标标签评价之前，先用 student/teacher 概率做 target-label-free 融合；
4. 固定最终预测后，再读取目标标签计算 accuracy、macro-F1、balanced accuracy 和混淆矩阵。

`ca_ccsr_wjdot_prior20` 使用 `teacher_safe_fusion.fusion_base: prior_balanced`。它先混合 teacher 与 student 概率，再根据目标域预测先验做平衡，目的是缓解最终预测类别塌缩：

```text
p_base = mix(student_probs, teacher_probs)
p_final ∝ p_base / observed_prior^prior_balance_strength
```

然后重新归一化得到最终概率。

该模块还会导出：

- `alpha_entropy_per_class.csv`
- `eta_distribution.csv`
- `teacher_student_disagreement.csv`
- `teacher_safe_fusion_summary.csv/json`
- per-class recall gain 等诊断结果。

### 4.8 论文可写成的贡献点

这部分可以写成：针对多源工况下“源域贡献随故障类别变化”的问题，提出 CoDATS 教师先验增强的类条件源域可靠性 WJDOT。该方法将逐源 OT 代价、源/目标原型距离、目标预测熵和源域类别错误率统一为类条件可靠性矩阵，动态控制不同源域在不同故障类别上的迁移贡献；同时通过 CoDATS 对抗对齐和 teacher-safe refinement 降低多源 OT 调整中的目标域漂移风险。

## 5. 三项改良点之间的逻辑关系

三项改良点可以组织成一条比较自然的论文叙事链。

第一层，`TPU-DeepJDOT` 解决“单源 OT 对齐不够稳”的问题。它不碰目标伪标签，只改善对齐代价和训练调度。

第二层，`CBTPU-DeepJDOT` 解决“目标域无标签样本如何安全利用”的问题。它在 TPU 稳定对齐基础上，加入三源一致性伪标签、强弱增强一致性和类别均衡。

第三层，`CA-CCSR-WJDOT` 解决“多源域每个类别应该相信哪个源域”的问题。它从单源 target reliability 扩展到多源 class-source reliability，并用 CoDATS teacher anchor 约束 refinement 不偏离稳定基线。

可以概括为：

```text
TPU: 让单源 OT 对齐更稳定
CBTPU: 让目标伪语义利用更保守、更均衡
CA-CCSR-WJDOT: 让多源迁移按类别选择可靠源域
```

## 6. Target label 使用边界

当前主线强调 target-label-free。代码中有多处约束和测试支持这一点。

- 单源 `deepjdot` 家族的 `compute_loss()` 接收 `target_batch` 时只使用 `target_x`，不使用 `target_y`。
- 多源 `ca_ccsr_wjdot` 的训练损失默认 `target_label_assist_weight: 0.0`。
- `src/evaluation/ca_ccsr_wjdot.py` 明确先用 student/teacher probabilities 固定最终融合预测，再读取 target eval labels 做指标。
- `tests/test_deepjdot.py` 检查 `tpu_deepjdot`、`cbtpu_deepjdot` 等方法改变 target labels 时 loss 不变。
- `tests/test_no_target_label_leakage.py` 检查 `ca_ccsr_wjdot` 等方法不会在训练 loss 中泄漏 target batch labels。

论文写作时建议明确说明：目标域标签只用于最终报告指标和混淆矩阵，不参与训练、选模、early stopping、门控校准或 teacher fusion。

## 7. 当前可以发给 chatbot 讨论的问题

1. 三项创新点是否应该在论文中分别命名为“时序原型非平衡 OT”“三源一致性置信伪标签”“类条件多源可靠性 WJDOT”，还是需要更短更学术的命名？
2. 单源两个方法是否应该写成一个递进章节，还是拆成两个小节：先 TPU，再 CBTPU？
3. 多源 CA-CCSR-WJDOT 是否应强调“类条件源域可靠性”作为核心贡献，还是强调“CoDATS teacher prior + WJDOT refinement”的工程闭环？
4. 实验表格中是否应该把 `target_ref` / `target_only` 明确标成 supervised reference，而不是 UDA baseline？
5. 如果某些场景中 `CBTPU` 不一定严格优于 `TPU`，论文应更强调平均趋势、目标样本接收率和类别塌缩缓解，还是继续调参追求严格单调？

## 8. 建议的论文表述草案

可以把当前成果压缩成如下表述：

本文针对流程工业故障诊断中的领域偏移问题，在 DeepJDOT/WJDOT 框架上设计了三项递进式改良。首先，提出时序-原型非平衡 DeepJDOT，通过源域类别原型、时序统计代价和非平衡最优传输增强单源迁移中的对齐稳定性。其次，提出置信均衡 TPU-DeepJDOT，综合 OT 传输语义、EMA teacher 预测语义和源域原型语义，仅在多证据一致时引入目标域伪监督，并通过类别均衡和一致性正则抑制伪标签噪声。最后，面向多源迁移提出 CoDATS 增强的类条件源域可靠性 WJDOT，构建源域-类别可靠性矩阵，动态选择不同故障类别下更可信的源域，并通过 CoDATS teacher anchor 和 final teacher-safe fusion 降低多源 refinement 的目标域漂移风险。

这一表述的重点是：三项改良都围绕同一个核心问题展开，即在无目标标签条件下，使工业时间序列故障诊断的跨域对齐更稳定、目标语义利用更保守、多源贡献分配更可靠。
