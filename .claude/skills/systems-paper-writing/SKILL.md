---
name: systems-paper-writing
description: 系统领域顶会(OSDI/SOSP/EuroSys/ATC/NSDI)论文写作指南。写作、修改、review系统方向论文的任何部分(abstract/intro/motivation/design/eval/related work)、画图、或按投稿checklist自查时使用。
---

# 系统领域顶会论文写作

写作或评审 OSDI/SOSP/EuroSys/ATC/NSDI 方向的论文时，先完整阅读本目录下的 [reference.md](reference.md)（由100篇顶会论文总结而成），然后严格按其中的结构与技巧执行。

核心原则速查（详见 reference.md）：

- 审稿人五维度：原创性、现实性、经验教训、设计选择、聚焦。写每一节前先问：这一节在哪个维度上加分？
- Intro 必须在第一页讲清：问题是什么、为什么难、已有方案为什么不够、我们的关键 insight 是什么、贡献列表。
- Motivation 用实测数据说话，不用形容词说话。
- Design 章聚焦新的部分，已知内容一笔带过；每个关键设计点都要写清"替代方案是什么、为什么不选"。
- Evaluation 必须回答 intro 里承诺的每一个 claim；end-to-end + ablation 缺一不可。
- 用户让你 review 论文草稿时，逐节对照 reference.md 中对应章节的要求和"投稿前 Checklist"给出具体、可操作的修改意见，引用原文行号。

写 HPC 会（SC/PPoPP/ICS/HPDC）论文时改用 `hpc-paper-writing` skill——两类会的叙事口味不同。
