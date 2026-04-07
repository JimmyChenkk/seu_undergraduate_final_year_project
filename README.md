# 领域自适应流程工业故障诊断研究工作区

本工作区用于开展 Tennessee Eastman Process (TEP) 领域自适应故障诊断研究，强调“代码、实验结果、论文”在同一 workspace 中协同推进。当前阶段处于第二阶段起点：先完成原始 pickle 的真实 schema 勘察、manifest 固化和 benchmark 元数据接口，再进入 baseline 与 DA 方法实现。

## 项目背景

- 研究主题：领域自适应流程工业故障诊断。
- 目标场景：当前主线聚焦 single-source to single-target 的通用领域自适应，后续如有必要再扩展到更多协议。
- 数据集：Tennessee Eastman Process Domain Adaptation，原始 `.pickle` 文件统一放在 `data/raw/`。
- 参考仓库：
  - `external/tep-domain-adaptation`：benchmark 与数据组织参考。
  - `external/TL-Fault-Diagnosis-Library`：方法实现参考。
  - `external/skada`：通用领域自适应工具箱参考。
  - `refs/algorithms/`：额外算法资料与论文实现参考。

## 重要约束

- 当前工作区内的 `external/` 与 `refs/` 默认视为只读参考区，不应随意修改其中已有文件。
- `data/raw/` 只存放原始数据，不要移动、重命名、删除原始数据文件。
- 新增内容优先放在 `src/`、`configs/`、`scripts/`、`paper/`、`runs/` 等工程目录中。

## 当前目录结构

```text
workspace/
├─ README.md
├─ .gitignore
├─ environment.yml
├─ requirements-benchmark.txt
├─ data/
│  ├─ raw/
│  ├─ benchmark/
│  └─ cache/
├─ src/
│  ├─ automation/
│  ├─ datasets/
│  ├─ methods/
│  ├─ trainers/
│  └─ evaluation/
├─ configs/
│  ├─ data/
│  ├─ method/
│  └─ experiment/
├─ scripts/
│  ├─ inspect_raw_data.py
│  ├─ build_benchmark.py
│  ├─ build_benchmark.sh
│  ├─ train.sh
│  ├─ eval.sh
│  └─ run_small_scale_round.sh
├─ paper/
│  ├─ thesis.tex
│  ├─ chapters/
│  ├─ figs/
│  └─ bib/
├─ external/
└─ refs/
   ├─ algorithms/
   ├─ papers/
   └─ reading/
```

## 数据放置约定

- 原始数据：放在 `data/raw/`
  - 预期为 6 个 `.pickle` 文件。
  - 当前将其视为“项目内不可变上游快照”，不要移动、重命名、删除。
- benchmark 元数据：放在 `data/benchmark/`
  - 当前阶段只生成 `manifest.json` 这类小型元数据，不复制大数组文件。
- 缓存或临时产物：放在 `data/cache/`

## 建议的研究推进路线

1. 先构建 benchmark：梳理 `data/raw/` 的字段格式、域划分、训练/验证/测试协议，并固化到 `data/benchmark/`。
2. 先跑 source-only baseline：建立最小可重复的上界/下界与日志记录规范。
3. 再接入 single-source UDA：优先选择实现稳定、文献成熟的方法做第一轮验证。
4. 采用 3 轮自主小规模优化：先批量跑简单场景，再由 Codex 自己读结果与图，继续改代码或配置复跑。
5. 最后整理图表并写论文：把 `runs/` 中的结果沉淀到 `paper/figs/` 与各章节草稿。

## Git 追踪策略

本项目的 Git 仓库直接建在 workspace 根目录，但只托管“核心代码和文档”，不托管大数据与训练输出：

- 会进入 Git：`src/`、`scripts/`、`paper/`、`README.md`、`environment.yml`、`requirements-benchmark.txt`
- 默认不进入 Git：`configs/`、`data/benchmark/`、`data/raw/`、`data/cache/`、`runs/`、`external/`、`refs/`、日志、临时文件

## 当前阶段常用命令

```bash
# 1) 首次创建基础环境
conda env create -f environment.yml

# 2) 激活项目环境
conda activate tep_env

# 3) 勘察 6 个原始 pickle 的真实结构
python scripts/inspect_raw_data.py

# 4) 安装 benchmark 训练依赖
pip install -r requirements-benchmark.txt

# 5) 生成 benchmark 元数据 manifest
bash scripts/build_benchmark.sh configs/data/te_da.yaml

# 6) 检查哪些文件会进入 Git
git status -sb
```

## 当前阶段范围

- 当前已经从“纯数据勘察”进入“小型 benchmark 复现准备”阶段。
- 方法范围先收敛为：`source_only`、`CORAL`、`DAN`、`DANN`、`CDAN`、`DeepJDOT`。
- 浅层 `JDOT` 基线已经移除；如需 OT 路线，统一使用 `DeepJDOT`。
- `external/` 与 `refs/` 继续只读；真正会进入 Git 的实验代码全部落在 `src/`、`configs/`、`scripts/`。
- 训练依赖目前还需要手动安装，项目代码不会擅自替你绕过缺包问题。
- 默认首轮小规模场景为：`mode1->mode4`、`mode4->mode1`、`mode2->mode5`、`mode5->mode2`、`mode3->mode6`、`mode6->mode3`。

## 复现命令示例

```bash
# single-source: source-only
bash scripts/train.sh \
  configs/data/te_da.yaml \
  configs/method/source_only.yaml \
  configs/experiment/benchmark_single_source.yaml

# single-source: DANN
bash scripts/train.sh \
  configs/data/te_da.yaml \
  configs/method/dann.yaml \
  configs/experiment/benchmark_single_source.yaml

# single-source: CDAN
bash scripts/train.sh \
  configs/data/te_da.yaml \
  configs/method/cdan.yaml \
  configs/experiment/benchmark_single_source.yaml

# 默认小规模首轮 sweep
bash scripts/run_small_scale_round.sh

# 参考仓库对齐的更长训练预算
bash scripts/run_small_scale_round.sh \
  --experiment-config configs/experiment/autonomous_small_scale_reference_budget.yaml

# 只跑一个方法和一个场景做 smoke test
bash scripts/run_small_scale_round.sh \
  --methods source_only \
  --scenes mode1:mode4 \
  --dry-run

# 汇总 runs/ 下按实验目录组织的结果
bash scripts/eval.sh runs
```

## 手动安装训练依赖

```bash
conda activate tep_env
pip install -r requirements-benchmark.txt
```
