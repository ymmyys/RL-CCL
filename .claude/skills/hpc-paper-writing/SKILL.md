---
name: hpc-paper-writing
description: 高性能计算顶会(SC/PPoPP/ICS/HPDC)论文写作指南。写作、修改、review HPC方向论文的任何部分(abstract/intro/performance analysis/optimization/eval/scaling实验)、画图、或按投稿checklist自查时使用。
---

# HPC 领域顶会论文写作

写作或评审 SC/PPoPP/ICS/HPDC 方向的论文时，先完整阅读本目录下的 [reference.md](reference.md)（由100篇顶会论文总结而成），然后严格按其中的结构与技巧执行。

核心原则速查（详见 reference.md）：

- HPC 论文的核心叙事是**性能**：让计算尽可能接近硬件极限，而不是提出新抽象。
- 与系统论文的关键区别：评判标准是性能提升显著性 + 性能理解深度；evaluation 重心是 scaling + roofline + breakdown，而非 end-to-end + ablation。
- 必须有 Performance Analysis / Motivation 章：实测找出瓶颈、量化离 peak 的距离，优化技术要与瓶颈一一对应。
- 深度绑定具体硬件特性是加分项不是减分项（写清楚在什么硬件上、利用了什么特性）。
- Scaling 实验（strong/weak scaling）在有意义的规模上做，说明并行效率损失来源。
- 用户让你 review 论文草稿时，逐节对照 reference.md 中对应章节的要求和"投稿前 Checklist"给出具体、可操作的修改意见，引用原文行号。

写系统会（OSDI/SOSP/EuroSys/ATC）论文时改用 `systems-paper-writing` skill——两类会的叙事口味不同。
