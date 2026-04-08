# 工作流

本文件约束当前工作区内 TEP 领域自适应 benchmark 的默认协作方式。原则只有三条：先小规模验证，再批量实验；先人工确认，再长时间 CUDA 跑；遇到异常先暂停，不擅自改环境。

## 1. 项目目标

- 当前阶段：benchmark 稳定化与论文骨架搭建阶段。
- 当前主线：先在 TEP DA 数据集上复现并比较已有 DA 方法，再进入改进方法设计与验证。
- 当前已落地范围：通用 `single_source` UDA；方法包括 `source_only`、`coral`、`dan`、`dann`、`cdan`、`deepjdot`；默认 backbone 以 `fcn` 为主。
- 后续扩展：当前版本先聚焦 FCN 主线，不保留其他 backbone 分支。
- 全量目标：先覆盖代表性单源单目标场景并形成稳定 benchmark，再开展改进方法实验。

## 2. 默认工作方式

1. 先查 `external/` 和 `refs/`，再把真正进入项目的代码写到 `src/`、`configs/`、`scripts/`。
2. 先跑小规模或代表性 slice，确认命令、输出路径、指标和图表链路正常。
3. Codex 默认先自己读结果与图，再做 `2~3` 轮有意义的小规模优化。
4. 达到检查点后再人工审查，再决定是否扩量。

- 每次操作前说明本轮目标、计划命令、预计写入目录。
- 每次产出后先展示关键指标和图表，再讨论是否扩量。
- 出现报错、CUDA 问题、缺包、权限问题、网络问题、环境改动需求、大规模写入或清理时，必须先暂停并询问。

## 2.1 默认自主循环

1. Round 1：先对 `source_only`、`coral`、`dan`、`dann`、`cdan`、`deepjdot` 跑首轮小规模 sweep。
2. Round 2：Codex 自己读取 `result.json`、`review.json`、混淆矩阵和两张 t-SNE，针对共享问题或高价值方法修改代码或配置后复跑。
3. Round 3：在同一组简单场景上确认优化是否稳定，不扩方法、不扩场景。
4. 默认人工检查点：首轮 sweep 完整，且至少完成 `1` 轮针对性优化复跑。
5. 普通退化不立即人工升级；只有硬异常才提前停止并汇报。

## 2.2 Codex Agent 执行环境要求

- Codex agent 不应假设自己会继承用户交互式终端里已激活的 `(tep_env)`；即使用户本地 prompt 已显示 `(tep_env)`，agent 新开的命令也可能实际落到系统 `python3`。
- 只要命令涉及 `python`、训练、评估、出图、自动化 sweep，Codex 默认显式使用 `conda run -n tep_env ...`，除非已经确认当前解释器就是 `tep_env` 内的 Python。
- 默认推荐写法：
  - `conda run -n tep_env python -m src.trainers.train_benchmark ...`
  - `conda run -n tep_env python -m src.automation.run_small_scale_round ...`
  - `conda run -n tep_env python -m src.evaluation.evaluate ...`
  - `conda run -n tep_env python -m src.evaluation.report_figures ...`
- 当前 `scripts/build_benchmark.sh`、`scripts/train.sh`、`scripts/eval.sh`、`scripts/export_figures.sh` 均复用统一环境解析逻辑；只要 `tep_env` 未激活但本机存在 `conda`，脚本会自动回落到 `conda run -n tep_env`。
- 若直接运行 Python 模块而不是脚本入口，Codex 仍应优先显式使用 `conda run -n tep_env ...`。
- 在首个 Python 类命令前，Codex 应优先确认实际解释器，例如运行 `conda run -n tep_env python -c "import sys; print(sys.executable)"`；如果 `conda` 不可用、`tep_env` 不存在、或环境损坏，先暂停并汇报，不自行改动全局环境。
- 涉及环境判断时，Codex 在汇报里应明确说明本次命令是否使用了 `conda run -n tep_env`，以及实际执行解释器路径。

## 3. 默认实验边界

- 单轮默认规模：`1 个方法 x 1 个场景 x 1 个 backbone x 1 个 fold`。
- 默认先沿用现有配置中的 epoch；是否统一放大到 `80 epoch`，等首轮结果稳定后再定。
- 默认方法：`source_only`、`coral`、`dan`、`dann`、`cdan`、`deepjdot`。
- 默认首轮简单场景：`mode1 -> mode4`、`mode4 -> mode1`、`mode2 -> mode5`、`mode5 -> mode2`、`mode3 -> mode6`、`mode6 -> mode3`。
- 默认代表性报告场景：`mode2 -> mode5`、`mode4 -> mode1`。
- 未经批准不要直接做：全量 `30` 场景 sweep、多方法并发长时间 CUDA 任务、明显扩大 `runs/` 体量的批量实验。

## 4. 输出与检查点

- 默认输出根目录：`runs/`。
- 单次运行目录命名：`runs/timestamp_method_scenario_backbone_fold/`。
- 单次运行目录下固定包含：`artifacts/`、`tables/`、`figures/`、`logs/`、`checkpoints/`。
- 批量或全量运行时，默认父目录形式为：`runs/timestamp_full_run/`。
- 批量父目录下包含：若干 `timestamp_method_scenario_backbone_fold/` 子目录，以及 `comparison_summary/` 目录；汇总表和对比图默认写到 `comparison_summary/tables/`、`comparison_summary/figures/`。
- 单次运行默认自动补齐：`tables/result.json`、`tables/review.json`、`figures/confusion_matrix.png`、`figures/tsne_domain.png`、`figures/tsne_class.png`。
- 批量运行默认自动补齐：`comparison_summary/tables/comparison.json`、`comparison_summary/tables/comparison.md`、`comparison_summary/tables/round_review.json`。
- 默认不保存 checkpoint，默认保留分析产物，即保持 `save_checkpoint: false`、`save_analysis: true`。
- 每次新方法首次跑通、新场景首次跑通、准备扩量、准备整理报告或提交 Git 前，都要先停下来检查。
- 每个检查点至少展示：
  - 关键指标 JSON
  - 基础对比图表
  - 混淆矩阵
  - 两类 t-SNE：分类聚合图、域融合图
- 结果目录命名保持参数化，至少包含时间、方法、setting、scenario。
- 旧结果只在完成汇总且确认不再引用后再手动清理。

## 4.1 默认判读顺序

1. 先看源域训练是否有效：`source_train_acc`。
2. 再看源域泛化是否可靠：`source_eval_acc`。
3. 再看目标域性能：`target_eval_acc`。
4. 再看混淆矩阵是否更接近对角线主导。
5. 再看 `tsne_domain` 是否体现域融合。
6. 再看 `tsne_class` 是否体现类别分散且清晰可分。

## 5. 工作区目录、文件与 Git 规则

当前 `notes/` 已删除，暂不作为固定目录。`external/`、`refs/`、`runs/` 下的文件默认都不 PUSH，因此这里主要罗列其余目录中的实际文件。以下判断只使用 `PUSH` 或 `不要 PUSH`。

### 5.1 根目录文件夹

| 目录 | 作用 | Git 建议 |
| --- | --- | --- |
| `.vscode/` | 本机编辑器配置 | 不要 PUSH |
| `configs/` | 数据、方法、实验协议配置 | PUSH |
| `data/` | 数据、缓存、benchmark 元数据 | 不要 PUSH |
| `external/` | 外部参考仓库 | 不要 PUSH |
| `paper/` | 论文内容与最终图表 | PUSH |
| `refs/` | 阅读资料与算法参考 | 不要 PUSH |
| `runs/` | 训练输出、图表、日志、汇总结果 | 不要 PUSH |
| `scripts/` | 数据处理、训练、评估脚本 |  PUSH |
| `src/` | 项目核心代码 |  PUSH |

### 5.2 根目录文件

| 文件 | 作用 | Git 建议 |
| --- | --- | --- |
| `AGENTS.md` | 工作区协作约束 | 不要 PUSH |
| `README.md` | 项目说明与使用入口 |  PUSH |
| `WORKFLOW.md` | 当前实验协作规则 | 不要 PUSH |
| `environment.yml` | Conda 环境定义 | PUSH |
| `requirements-benchmark.txt` | benchmark 训练依赖 | PUSH |

### 5.3 `configs/` 与 `data/` 下的文件

以下只说明文件作用，是否 `PUSH` 按 `5.1` 的上级目录判断。

| 文件 | 作用 |
| --- | --- |
| `configs/data/te_da.yaml` | 数据路径与协议基础配置 |
| `configs/experiment/benchmark_single_source.yaml` | 单源 benchmark 主实验配置 |
| `configs/experiment/autonomous_small_scale.yaml` | 默认自主小规模 sweep 配置 |
| `configs/experiment/full_36_fcn_aggressive_5090.yaml` | 当前全量单源加多源主跑配置 |
| `configs/experiment/report_s2_t5.yaml` | 代表性报告场景配置 |
| `configs/experiment/report_s4_t1.yaml` | 代表性报告场景配置 |
| `configs/method/source_only.yaml` | source-only 方法配置 |
| `configs/method/coral.yaml` | CORAL 方法配置 |
| `configs/method/dan.yaml` | DAN 方法配置 |
| `configs/method/dann.yaml` | DANN 方法配置 |
| `configs/method/deepjdot.yaml` | DeepJDOT 方法配置 |
| `configs/method/cdan.yaml` | CDAN 方法配置 |
| `data/benchmark/manifest.json` | benchmark 元数据清单 |
| `data/benchmark/.gitkeep` | 空目录占位文件 |
| `data/cache/.gitkeep` | 空目录占位文件 |

### 5.4 `paper/` 与 `scripts/` 下的文件

以下只说明文件作用，是否 `PUSH` 按 `5.1` 的上级目录判断。

| 文件 | 作用 |
| --- | --- |
| `paper/thesis.tex` | 论文主入口 |
| `paper/bib/references.bib` | 参考文献库 |
| `paper/chapters/01_intro.tex` | 绪论章节 |
| `paper/chapters/02_related_work.tex` | 相关工作章节 |
| `paper/chapters/03_method.tex` | 问题定义与 benchmark 平台章节 |
| `paper/chapters/04_experiments.tex` | 基线复现与 benchmark 分析章节 |
| `paper/chapters/05_improved_method.tex` | 改进方法设计与实验验证章节 |
| `paper/chapters/06_conclusion.tex` | 总结与展望章节 |
| `scripts/build_benchmark.py` | 生成 benchmark 元数据 |
| `scripts/build_benchmark.sh` | benchmark 构建脚本入口 |
| `scripts/train.sh` | 训练脚本入口 |
| `scripts/eval.sh` | 结果汇总脚本入口 |
| `scripts/export_figures.sh` | 图表导出脚本入口 |
| `scripts/inspect_raw_data.py` | 原始数据勘察脚本 |
| `scripts/run_small_scale_round.sh` | 默认小规模自主 sweep 入口 |

### 5.5 `src/` 下的文件

以下只说明文件作用，是否 `PUSH` 按 `5.1` 的上级目录判断。

| 文件 | 作用 |
| --- | --- |
| `src/__init__.py` | 包入口 |
| `src/backbones/__init__.py` | backbone 注册入口 |
| `src/backbones/fcn.py` | FCN backbone 实现 |
| `src/datasets/__init__.py` | 数据模块入口 |
| `src/datasets/te_da_dataset.py` | TEP DA 元数据与接口 |
| `src/datasets/te_torch_dataset.py` | Torch 数据准备逻辑 |
| `src/evaluation/__init__.py` | 评估模块入口 |
| `src/evaluation/evaluate.py` | 结果汇总逻辑 |
| `src/evaluation/review.py` | 单次与批量评审摘要逻辑 |
| `src/evaluation/report_figures.py` | 报告图表导出逻辑 |
| `src/losses/__init__.py` | 损失模块入口 |
| `src/losses/domain.py` | 领域自适应损失实现 |
| `src/methods/__init__.py` | 方法注册入口 |
| `src/methods/base.py` | 方法基类 |
| `src/methods/source_only.py` | source-only 实现 |
| `src/methods/coral.py` | CORAL 实现 |
| `src/methods/dan.py` | DAN 实现 |
| `src/methods/dann.py` | DANN 实现 |
| `src/methods/deepjdot.py` | DeepJDOT 实现 |
| `src/methods/cdan.py` | CDAN 实现 |
| `src/automation/run_small_scale_round.py` | 默认小规模 sweep 执行逻辑 |
| `src/trainers/__init__.py` | 训练模块入口 |
| `src/trainers/train_benchmark.py` | benchmark 训练主逻辑 |

### 5.6 当前默认结论

- `PUSH / 不要 PUSH` 以 `5.1` 和 `5.2` 为准。

### 5.7 Git 最小命令集合

以下命令用于“本地 `.git/` 已删除后，重新初始化当前项目，但只提交 `.gitignore` 允许提交的内容”。

#### 首次重新接回远端

```bash
git init
git branch -M main
git add .
git status
git diff --cached --name-only
git commit -m "chore: initial import"
git remote add origin git@github.com:JimmyChenkk/seu_undergraduate_final_year_project.git
git push -u origin main
```

#### 日常最常用

```bash
git status
git add .
git diff --cached --name-only
git commit -m "feat: xxx"
git push
```

#### 防止误传时优先使用

```bash
git status --ignored
git add -n .
git check-ignore -v <文件或目录>
git restore --staged <文件或目录>
```

#### 说明

- `git add .` 会自动跳过 `.gitignore` 中已忽略的内容。
- `git diff --cached --name-only` 用于确认“这次 commit 到底会包含哪些文件”。
- 如果只想撤回暂存而不删除文件，使用 `git restore --staged <文件或目录>`。

## 6. 安全规则

- 不自行修复系统级 CUDA、驱动、`conda`、全局包问题。
- 默认不主动安装包；确需安装时先说明用途、影响范围和命令。
- `external/`、`refs/`、`data/raw/` 默认视为只读区。
- 涉及大规模删除、移动、重命名、清理时必须先确认。
- 如果被外部因素阻塞，先汇报，不为了绕过问题而临时改项目方向。

## 7. 当前下一步

1. 确认全量 sweep 的触发条件、`runs/` 体量上限、`80 epoch` 何时启用。
2. 先用 `fcn` 跑完 `6` 个简单双向单源场景上的首轮 sweep。
3. 让 Codex 先完成至少 `1` 轮自主优化复跑，再进入人工审查节点。
