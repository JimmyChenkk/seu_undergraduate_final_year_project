# 网页 Chatbot 文献检索提示词

说明：
- 这些提示词不进入正式论文，只作为后续补文献、补表述时的辅助脚手架。
- 让网页 Chatbot 返回结果时，优先要求给出论文标题、年份、作者、核心观点、是否与工业过程故障诊断直接相关。
- 所有新增引用都需要手工二次核对，再决定是否写入 `paper/bib/references.bib`。

## 1. TEP benchmark 与跨工况诊断

```text
请帮我检索 Tennessee Eastman Process（TEP）领域自适应/跨工况故障诊断相关论文，重点找 2020 年以后与 benchmark、统一实验协议、跨模式迁移设置有关的工作。请输出：
1. 论文标题、作者、年份、发表 venue
2. 论文是否使用 TEP 或 TEP domain adaptation 数据集
3. 论文的核心任务设置（单源/多源/有无目标域标签）
4. 论文最值得引用的 2-3 句结论
5. 如果只让我在本科论文里补 3 篇最相关文献，你推荐哪 3 篇，为什么
```

## 2. 教师--学生一致性与时序数据

```text
请帮我检索“teacher-student consistency / mean teacher / consistency training”在时间序列分类、故障诊断或工业过程迁移中的代表性论文。请重点关注：
1. 为什么 EMA teacher 比直接用学生当前预测更稳定
2. 弱增强/强增强在时间序列任务中常见的做法
3. 这些方法在工业故障诊断里有哪些限制
请给出适合放进本科论文第三章的 5 个高质量论述点，每个点配 1-2 篇参考文献。
```

## 3. 伪标签可靠性与样本筛选

```text
请帮我检索无监督领域自适应或半监督学习中“pseudo-label reliability / confidence filtering / curriculum pseudo labeling / class-balanced pseudo label selection”相关工作。我要写工业过程故障诊断论文的“可靠性门控”章节，请输出：
1. 单一 confidence threshold 的常见缺陷
2. 多因素 reliability score 的常见设计思路
3. class-wise selection / class-balanced selection 的意义
4. 如果我要把这些内容写成一章，最稳妥、最不夸大的表达方式是什么
```

## 4. 原型方法与类条件结构

```text
请帮我检索 prototype-based unsupervised domain adaptation、class-conditional alignment、prototype memory 在迁移学习中的代表性工作，尤其关注：
1. prototype memory 为什么能缓解 class confusion
2. source prototype 与 target prototype 各自的作用和风险
3. prototype attraction / separation loss 常见的写法
4. 哪些说法适合工业时序故障诊断论文，哪些说法会显得夸大
请尽量输出适合写进第五章的中文总结框架。
```

## 5. 多源迁移与更强创新点

```text
请帮我检索 multi-source domain adaptation 在工业过程故障诊断或时间序列迁移中的代表性工作，重点看 source weighting、source reliability、source-target relation modeling。请输出：
1. 现有多源方法的关键思想
2. 如果我当前已有一个 reliability-gated + prototype-based 的单/多源通用框架，最自然的升级方向是什么
3. 哪些 upgrade 既有研究价值，又适合本科毕设范围
```

## 6. 流程工业先验与结构化迁移

```text
请帮我检索工业过程故障诊断中结合 process prior、control loop、variable graph、causal relation 或 knowledge-guided representation 的工作。我的目标是思考如何把当前纯数据驱动的 RCTA 升级为“带工业先验的结构化迁移方法”。请输出：
1. 最常见的工业先验类型
2. 这些先验如何与深度迁移学习结合
3. 哪些方向最适合在 TEP 数据集上做实验验证
```
