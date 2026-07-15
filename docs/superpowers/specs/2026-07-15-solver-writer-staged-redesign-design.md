# 求解器与写作器分阶段重设计

- **日期**：2026-07-15
- **状态**：已通过设计评审，待编写实现计划
- **范围**：`modeling-agent` 项目的 `solver`（求解器）与 `writer`（写作器）两个 Agent 的内部重设计；编排器、其余 Agent 结构不动
- **目标主题**：提高求解正确率；让求解器写更多、更周全的代码；让写作器输出更充实的论文；并保证论文与上游分析/求解结果一致、不跑题

---

## 1. 背景与目标

### 1.1 用户诉求

1. 大模型求解问题时**正确率**不足，是本次改进的首要目标。
2. 代码智能体（solver）写的代码**太少、考虑不周**；希望**分阶段写更多代码**，用 Matplotlib 画图，把问题求解做得更细致。
3. 写作智能体（writer）写的论文**内容太少**；希望和代码智能体一样，想办法**多输出内容**。
4. 论文内容必须与前面的分析、求解结果**一致**，不得跑题、不得编造。

### 1.2 现状与问题（基于代码阅读）

- **求解器**（`app/agents/solver.py`）：一次 LLM 调用生成**一个** ≤200 行的单体脚本，然后 `py_compile` -> `subprocess` 执行（cwd=任务根，超时 60s）-> 失败则把整段错误贴回去让 LLM **重新生成整段脚本**，最多 3 次。问题：单体脚本浅、阶段不分、修复会丢失已正确部分、无结果合理性校验、无随机种子/无 Agg 后端。
- **分析师**（`analyst.py`）已在 `analysis.md` 输出"问题拆解"，**建模师**（`modeler.py`）已按子问题分别建模——但**求解器忽略了这一结构**，仍写一个脚本。这是"代码太少、考虑不周"的根因。
- **写作器**（`writer.py`）：**一次** LLM 调用生成整篇论文，受 `max_tokens: 8192` 上限约束。一次调用无法产出长而细致的论文——这是"内容太少"的根因。
- **审查器**（`reviewer.py`）已有硬规则：求解失败则 `solving≤30` 并回退 `solver`；论文数值在求解输出中找不到则视为编造扣分。本设计在产稿/产结果前就把这类问题拦住，减少审查回退。

### 1.3 选定方案

在三个候选方案中选定**方案二（推荐）**：规划 + 按子问题×按阶段流水线 + 写作分节。三者均含 Matplotlib 画图、分层校验、写作分节；方案二在正确率收益与复杂度/成本间最平衡。校验方式选定**分层校验**（程序化硬检查 + LLM 自查），通用、不依赖标准答案。

---

## 2. 总体架构与流程

**核心原则：编排器不动，改动收敛在求解器和写作器内部。**

五个 Agent 顺序（`analyst -> modeler -> solver -> writer -> reviewer`）、三种运行模式、检查点、审查回退循环全部保持不变。求解器与写作器各自从"一次调用"变成"内部多步流水线"。审查器的硬否决（求解失败 -> 回退 `solver`）依旧生效。

### 2.1 子问题来源

分析师已在 `analysis.md` 写了"问题拆解"，建模师也按子问题建模。求解器的**规划步**用一次 LLM 调用，综合 analyst + modeler 产物，产出结构化求解计划（不改动 analyst/modeler，把"读懂子问题"的职责放在求解器内）。若规划步判断只有一个子问题，退化为单子问题流水线，行为兼容现状。

### 2.2 求解器内部流程（3 步）

```
规划(plan) -> 逐子问题×逐阶段执行(execute) -> 汇总自查(self-critique)
```

### 2.3 写作器内部流程（4 步，含一致性校验）

```
大纲(outline) -> 逐节生成(per-section) -> 拼接+扩写薄节(assemble) -> 一致性校验(consistency)
```

### 2.4 整体效果

求解器从 1 个脚本变成"子问题数 × 阶段数"个小脚本（代码量与周密度自然上升）；写作器从 1 次调用变成"大纲 + 各节 + 扩写 + 一致性"次调用（内容自然充实）。两者均为"生成 -> 校验 -> 有界返修"，机制对称。

---

## 3. 求解器重设计

### 3.1 规划步（plan）

- **输入**：题目 + `analysis.md` + `model.md` + 数据文件清单。
- 一次 LLM 调用，强约束输出 JSON 计划：

```json
{
  "subproblems": [
    {
      "id": "sub1",
      "title": "...",
      "goal": "...",
      "stages": [
        {"name": "data",   "goal": "...", "input_files": [], "output_file": "data.csv", "method": "...", "figures": []},
        {"name": "model",  "goal": "...", "input_files": ["data.csv"], "output_file": "model.pkl", "method": "...", "figures": []},
        {"name": "solve",  "goal": "...", "input_files": ["model.pkl"], "output_file": "result.json", "method": "...", "figures": [], "expected_range": null},
        {"name": "analyze","goal": "...", "input_files": ["result.json"], "output_file": "analysis.json", "method": "...", "figures": []},
        {"name": "plot",   "goal": "...", "input_files": ["result.json"], "output_file": "", "method": "...", "figures": ["sub1_1_curve.png"]}
      ]
    }
  ]
}
```

- 阶段名从固定调色板选：`数据 / 建模 / 求解 / 分析 / 画图`，每个子问题按需取子集（至少含"求解"，推荐含"画图"）。
- 落盘 `artifacts/solution/plan.json`。
- JSON 解析失败：重试 1 次；仍失败则退化为单子问题单阶段（=现状），绝不因规划硬崩。

### 3.2 执行步（execute）：逐子问题、逐阶段

- 每个子问题有独立工作目录 `artifacts/solution/<sub_id>/`，作为该子问题所有阶段脚本的 `cwd`；阶段间靠目录里的中间文件（`data.csv`/`model.pkl`/`result.json`…）自然交接，状态清晰、可单独重跑。
- 阶段脚本按序命名 `01_data.py`、`02_model.py`…。
- 每阶段流程：生成代码 -> `py_compile` 语法检查 -> 执行（cwd=子问题目录，超时 `stage_execution_timeout`，默认 120s）-> 分层校验（见第 4 节）-> 失败则带错误重生成（有界重试）。
- **STAGE_RESULT 约定**：每阶段脚本末尾打印一行 `STAGE_RESULT: {json}`，含 `ok / metrics / files / figures / notes`，框架解析它供校验与下游使用（类似现有 `FIGURES:` 约定）。
- **画图**：专设"画图"阶段（"分析"阶段也鼓励画）；强制 Agg 后端、英文标签、`savefig+close`，保存到 `artifacts/figures/<sub_id>_<n>_<desc>.png`。
- **可复现性前导**：框架给每个脚本统一注入前导（`random.seed(42)`、`numpy.random.seed(42)`、`matplotlib.use('Agg')`），不依赖 LLM 自觉，提升确定性与正确率。种子可由 `preamble_seed` 配置。

### 3.3 汇总自查步（self-critique）

- 聚合所有阶段的 `STAGE_RESULT` + stdout + 图表 + 失败阶段，一次 LLM 调用做一致性/正确性复查（子问题答案是否自洽、单位/假设是否一致、有无红旗）。
- 写聚合 `output.txt`（各子问题关键结果）、`status.json`（`executed` = 所有关键阶段成功，仍驱动审查器硬否决）、`summary.md`。

### 3.4 关键兼容点

- `output.txt` / `status.json` 路径与名称不变，编排器 `_build_ctx` 与写作器/审查器读取方式完全不受影响。
- `ctx.solution_stdout / executed / error / figures` 照旧设置。
- `solve.py` 单文件产物取消，改为多脚本目录；`write_artifact(solver)` 改为写 solution 目录，`artifact_path="artifacts/solution"`（本就是目录）。需一个小 store 辅助方法。

### 3.5 求解器存储布局

```
artifacts/solution/
  plan.json  status.json  output.txt  summary.md
  <sub_id>/  01_data.py  02_model.py …  (中间文件由 LLM 管理)
artifacts/figures/<sub_id>_<n>_<desc>.png   (扁平，沿用 list_figures)
```

---

## 4. 分层校验机制

每个阶段执行后，先过**程序化硬检查**，再过 **LLM 自查**；两层都过才进入下一阶段。无标准答案时无法保证结果一定正确，校验目标是抬高正确率地板：拦住崩溃、非法值、方法性红旗、阶段间不一致。

### 4.1 第一层：程序化硬检查（确定性、零额外调用）

- `exec_ok`：退出码 0、未超时。
- `result_line`：能解析到 `STAGE_RESULT:` 且 `ok=true`。
- `output_file`：计划声明的输出文件存在且非空。
- `finite`：`metrics` 里的数值全部有限（无 NaN/Inf）；若计划给了 `expected_range` 则校验落在范围内。
- `figures`：画图阶段，`STAGE_RESULT.figures` 里每个文件在 `figures_dir` 真实存在、非 0 字节、且为 PNG。
- `shape`（可选）：`metrics` 上报的维度与计划预期一致。

### 4.2 第二层：LLM 自查（语义、一次额外调用）

- **输入**：阶段目标（来自 plan）+ 代码 + `STAGE_RESULT` 指标 + stdout 摘要。
- **判断**：输出是否合理回应阶段目标、方法是否恰当、有无红旗、与上游是否一致。
- **输出**：JSON `{passed, issues, suggestion}`。
- 拦"跑通了但方法错"这类硬检查抓不到的问题。可用更快/更便宜模型跑（`self_critique_model` 可配，默认同模型）。

### 4.3 门控与有界重试

- 硬检查失败 -> 当作执行错误 -> 带错误重生成（执行类重试，上限 `max_stage_retries`，默认 2 次）。
- 自查不通过 -> 把 `issues` 回填 -> 再重生成（自查类重试，上限 `max_critique_retries`，默认 1 次）。
- 单阶段总重生成上限 3 次（`max_regen_per_stage`）；耗尽则标记该阶段失败，但**不中断整任务**——其他子问题继续求解，聚合 `status` 记录哪些阶段失败。

### 4.4 与审查器衔接

- `status.json.executed` 定义为：**所有子问题的"求解"关键阶段都成功**（非关键阶段如画图失败不致整体失败）。
- 审查器现有硬规则（求解失败 -> `solving≤30` -> 回退 `solver`）照常生效；同时给审查器喂**逐子问题结果 + 阶段状态**，让它能判断"跑通了但结果对不对"，而不只是"跑没跑"。
- 汇总自查（3.3）做跨子问题一致性，是聚合层的第二道。

### 4.5 诚实说明

分层校验能显著抬高地板（崩、非法值、方法红旗、不一致都会被拦），但**无标准答案时无法保证结果一定正确**——这是选定"通用、不依赖标准答案"的取舍。若日后某题有参考答案，可再叠加数值比对（本次不纳入）。

---

## 5. 写作器重设计

### 5.1 大纲步（outline）

- **输入**：题目 + `analysis.md` + `model.md` + **逐子问题求解结果**（聚合 `output.txt` + 各子问题 `STAGE_RESULT`）+ 图表 + 各子问题代码。
- 一次 LLM 调用产出结构化大纲：9 个固定章节（摘要 / 问题重述 / 问题分析 / 模型假设 / 符号说明 / 模型建立与求解 / 模型评价与推广 / 参考文献 / 附录），每节给出**要点、目标篇幅、引用哪些子问题结果/图表/代码**；"模型建立与求解"按子问题拆子节。
- 落盘 `artifacts/paper/outline.json`。

### 5.2 逐节生成步（per-section）

- **每节一次 LLM 调用**，只喂该节相关上下文（摘要喂全局摘要；某子问题求解子节只喂该子问题的模型+结果+图表+代码；附录喂全部代码）。
- 每节单次 token 预算更充裕（如 4096~8192/节），因为调用范围聚焦 -> 每节都能写深写透。这是"内容变充实"的关键。
- 图表按需嵌入：相关节被告知用 `![图N 说明](figures/<文件>)` 嵌入并解读，沿用现有规则但落到节级。
- 各节落盘 `artifacts/paper/sections/NN_<title>.md`。

### 5.3 拼接 + 扩写薄节步（assemble + expand）

- 按序拼接成 `paper.md`。
- 检测过薄节：低于最小字数阈值（`min_section_chars` 可配，如摘要<400、求解子节<600）或缺关键要素（某子问题无结果引用）。
- 对每个薄节做一次扩写调用（给当前文本 + 相关上下文 + "扩写到目标篇幅/深度"），有界（`max_expand_sections`，默认最多 4 节）。重组装成 `paper.md`。

### 5.4 一致性 / 不跑题保障（consistency check）

把"忠实于上游"做成写作器流水线里的显式校验层（组装后、终稿前），与求解器的"硬检查 + LLM 自查"对称：

- **A. 逐节接地（grounding，写入每节 prompt）**：每节调用强制约束——只能用所提供的上游产物（`analysis.md`/`model.md`/求解结果）中的事实与数值；不得编造方法、结果或数字；所需数值若不在给定材料中，须如实标注而非杜撰。因为每节只喂相关上下文，模型手上有真实素材，接地才有依据。
- **B. 一致性校验步**：
  - **程序化数值核对（确定性）**：从各节抽取所有数字，逐一核对是否出现在求解输出（`output.txt` / 各子问题 `STAGE_RESULT` 指标）中；对不上的标为疑似编造。
  - **LLM 语义一致性（一次调用）**：把组装稿与 `analysis.md`+`model.md`+求解结果对照，检查：方法/模型是否与建模一致、结论是否有结果支撑、是否跑题夹带无关内容、有无过度推断。输出 JSON `{offending_sections:[{section, issues}], off_topic, fabricated_numbers}`。
- **C. 有界返修**：对命中的节，带"一致性问题"回填重生成（每节上限 1 次、总计≤3 节），再重组装；仍不通过的在 `paper.md` 标注"待人工核对"，不中断整篇。
- **D. 与审查器呼应**：审查器已有"数值对不上=编造扣分"规则；现在写作器在**产稿前**就先自检，把跑题/编造拦在前面，减少审查回退。

### 5.5 写作器兼容性与存储

- 最终产物仍是 `artifacts/paper.md`（`ARTIFACT_NAMES` 不变），审查器读取方式不变；`ctx.figures` 不变；编排器不受影响。中间产物进 `artifacts/paper/`（新增子目录）。
- `postprocess` 仍返回字数/章节数摘要。

```
artifacts/
  paper.md                 (最终，路径不变)
  paper/                   (新增，中间产物)
    outline.json
    sections/01_abstract.md … 09_appendix.md
```

### 5.6 写作器成本与容错

- **成本**：从 1 次调用变成约 1(大纲)+9(各节)+N(扩写≤4)+1(一致性) ≈ 12~15 次，但每次聚焦，总产出远比单次充实。
- **容错**：单节调用失败重试 1 次；仍失败则写占位说明该节待人工补，不中断整篇，拼出已成功部分。

---

## 6. 数据流 / 存储 / 状态

### 6.1 数据流

- 求解器内部：plan -> 逐子问题×逐阶段 -> 汇总自查；对外 `ctx.solution_stdout/executed/error/figures` 不变。
- 写作器内部：outline -> 逐节 -> 拼接+扩写 -> 一致性校验 -> 终稿；对外产物 `paper.md` 不变。
- **编排器零改动**：`_build_ctx` 仍读 `output.txt`/`status.json`（路径名不变）；`AGENT_ORDER`、检查点、审查回退循环原样保留。审查器输入更丰富（喂逐子问题结果+阶段状态），但读取方式不变。

### 6.2 存储/状态变更（纯文件，无 schema 迁移）

- **新增**：`artifacts/solution/plan.json`、`<sub_id>/*.py`+中间文件、`summary.md`；`artifacts/paper/outline.json`、`paper/sections/*.md`。
- **保留**：`output.txt`、`status.json`、`paper.md`、`figures/` 名称路径不变。
- **取消**：`solve.py` 单文件产物（改为多脚本目录）；新增 store 辅助方法写 solution 目录。
- `state.json` / `AgentRecord` 结构不变；阶段细节走 `logs/solver.jsonl`、`logs/writer.jsonl` 的新事件类型（`plan` / `stage_start` / `stage_done` / `verify` / `consistency`）供前端流式展示。

---

## 7. 错误处理

- 阶段执行/硬检查失败 -> 有界重生成（执行类 2 次）；耗尽标记该阶段失败，**不中断**，其他子问题继续。
- 自查不通过 -> 1 次返修；仍不过则接受但标"低置信"。
- 某子问题"求解"关键阶段全失败 -> 聚合 `executed=False` -> 审查器硬否决 -> 回退 `solver`（沿用现有循环）。
- 非关键阶段（如画图）失败 -> `executed` 仍 True，记录缺失图，写作器标注。
- 规划步 JSON 解析失败 -> 重试 1 次；仍失败则退化为单子问题单阶段（=现状），绝不因规划硬崩。
- 写作单节失败重试 1 次；仍失败写占位；一致性校验不可修复的节标"待人工核对"。所有重试有界，流水线必终止。

---

## 8. 测试（pytest）

- **单元**：`STAGE_RESULT` 解析、前导注入、各硬检查函数（有限值/输出文件存在/图有效 PNG/范围）、数值抽取与交叉核对（写作器一致性）、store 辅助方法。
- **集成**：以 **2025 A 题（3 子问题）**为验证用例跑全流程，核对 plan 含 3 子问题、各产出结果+图、论文嵌图、一致性通过、审查评分合理；另测 1 子问题兼容退化、故意报错触发修复重试 + `executed=False` + 审查回退。
- **前端冒烟**：新日志事件类型能在前端流式渲染（仅验证不破坏，不做 UI 重构）。
- **兼容**：验证退化路径下下游（writer/reviewer）行为不变。

---

## 9. 配置新增（config.yaml）

- `solver:` 段新增：`max_stage_retries: 2`、`max_critique_retries: 1`、`max_regen_per_stage: 3`、`stage_execution_timeout: 120`、`preamble_seed: 42`、`self_critique_model: null`。
- 新增 `writer:` 段：`min_section_chars: {abstract: 400, solving_sub: 600}`、`max_expand_sections: 4`、`consistency_check: true`、`consistency_model: null`。
- 默认值保证开箱即用的合理行为；`*_model: null` 表示继承该 Agent 当前模型。

---

## 10. 范围

### 10.1 纳入

求解器分阶段 + 分层校验、写作器分节 + 一致性校验、存储辅助方法、配置新增、日志事件、测试、A 题验证。

### 10.2 不纳入（YAGNI）

- 独立对抗式校验 Agent（方案三）、标准答案比对、失败重规划、Docker 沙箱（另立议题）。
- 改动 `analyst` / `modeler` / `reviewer` / `orchestrator` 的结构（仅丰富审查器输入，无结构变更）。
- 前端 UI 重构（仅保证新日志事件能渲染）。

### 10.3 向后兼容

旧任务仍可加载；单脚本路径作为退化保留。

---

## 11. 工作流约定：GitHub 备份

- 实现阶段**每次改完代码都 `commit + push` 到 GitHub 做备份**。
- 当前工作区非 git 仓库；实现开始时先 `git init`、设置 remote（由用户确认：网页建空仓库给地址，或用 `gh` CLI 新建），之后每次代码修改后提交推送。
- 本设计文档作为首个本地提交先行纳入版本管理；待 remote 就绪后一并推送。

---

## 12. 后续

设计评审通过后，交由 **writing-plans** 技能编写详细实现计划，再进入实现。
