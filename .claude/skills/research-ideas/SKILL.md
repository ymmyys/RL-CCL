---
name: research-ideas
description: 科研选题与idea挖掘。当用户想找research idea、判断某个idea是否已被做过(prior-art/novelty检查)、调研某方向的开放问题与瓶颈、或对候选idea做可行性评估时使用。面向系统/HPC方向(OSDI/SOSP/EuroSys/ATC/SC/PPoPP/ICS/HPDC)。
---

# Research Idea 挖掘与验证

这是一个多阶段深度调研流程。**不要凭模型记忆直接回答**——顶会 idea 的生死取决于最近 6-12 个月的 arXiv 论文，必须大量使用 WebSearch/WebFetch 实时检索。使用 Workflow 工具做多智能体并行调研（用户已开启 ultracode）。

## 流程

### 阶段1：多视角扫描（并行 agent，每个视角一个）
- **框架/系统全景**：该方向现有的所有系统与论文，各自架构、解决的瓶颈、venue、年份。
- **prior-art 检查**：对用户已有的每个候选 idea，用 3-4 组不同关键词（中英文）搜 arXiv 2025-2026 新论文，评估拥挤度。
- **开放问题**：论文/技术博客/工业界报告点名承认但论文很少的瓶颈，按"论文最少但痛点最大"排序。
- **venue 适配**：目标会议（系统会 vs HPC 会）最近两年收了什么相关论文，口味差别。

### 阶段2：综合生成候选 idea（10个左右）
每个 idea 必须包含：问题陈述（瓶颈是什么+证据）、why now、系统设计切入点、eval 方案（workload/baseline/指标/GPU规模）、最接近的已有工作、差异化、适合的 venue、上手难度与风险。要覆盖"稳妥的第一篇"到"有野心的大 idea"的梯度，并至少含一个低资源可做的（测量研究/trace分析/性能建模/仿真器）。

### 阶段3：对抗性新颖性验证（每个 idea 一个 agent）
每个候选 idea 交给一个"苛刻审稿人" agent，默认立场"这个 idea 已经被做过了"，努力搜索证伪；找到竞品则给出恢复新颖性的调整方案（fix）。

### 阶段4：输出
按（新颖性 × 可行性 × venue 适配）排序给出最终推荐，明确标注每个 idea 的拥挤度（low/medium/high）、最接近竞品、和建议的差异化切口。评估 novelty 时对照本目录同级的 `systems-paper-writing` / `hpc-paper-writing` skill 中审稿人评判标准。

## 原则
- 每条结论注明来源（论文名/系统名/链接/年份），搜不到竞品也要说明搜了哪些关键词。
- 工业界技术博客（ByteDance/Moonshot/阿里/NVIDIA/Meta/DeepSeek 等）和开源框架的 issue/design doc 是 idea 金矿，与 arXiv 同等对待。
- idea 的"拥挤度"比"绝对新颖"更重要：一个方向 2025 年出了 8 篇 arXiv，即使都没中会，novelty 空间也已经很小。
