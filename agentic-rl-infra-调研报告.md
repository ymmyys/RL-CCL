# Agentic RL Infra 选题调研最终报告

*2026-07-23 ~ 07-24，三轮多智能体调研+对抗验证，约 285 万 token 搜索，24 个调研/验证 agent*

## 一、总览：候选 idea 排序

| # | Idea | 拥挤度 | 裁决 | 资源门槛 | 与你背景的匹配 |
|---|------|--------|------|----------|----------------|
| A | 跨框架 infra 级故障注入 measurement | medium | ✅ 修正后可行（先行） | 8卡2-3周 | ★★★（fault-inject 直接复用） |
| B | 轨迹恢复跨层一致性协议运行时 | high | ✅ 收窄到协议层可行 | 中等 | ★★★（容错履历） |
| 1 | RMcast：训推权重 reshard-broadcast 形式化为 collective 原语 | medium | ✅ 条件通过，需 go/no-go 实验 | 64卡+仿真 | ★★★（通信库主场） |
| 3 | RL-Calculon：agentic RL 解析性能模型 | medium | ✅ 条件通过，需重定位 | 8-32卡 | ★★（PMBS 8月可占坑） |
| 6 | SandFS：Lustre 上的 sandbox 供给存储层 | medium | ✅ 条件通过 | 校级超算即可 | ★★（组内可行性最好之一） |
| 8' | VeriServe 修正版：可信 reward 服务（故障域区分） | — | ✅ 修正路线与 A/B 天然串联 | 中等 | ★★★ |
| 2 | RL-Traffic Orchestrator：单 job 内异质流量干扰+调度 | high | ⚠️ 条件通过但窗口收窄快 | 需共享 RoCE 环境 | ★★★ |
| 4 | AgentRL-Sim：闭环仿真器 | high | ⚠️ RLSim 已占位，须重定位 | 8-32卡 | ★★ |
| 5 | AuroraRL：超算/AMD/DCU 上的 agentic RL | medium | ⚠️ 硬前置：512卡 AMD/DCU 机时 | 极高 | ★★（RCCL 专长可用） |
| 7 | TrajStore：RDMA 原生 experience store | high | ⚠️ 核心卖点已被 TransferQueue+NIXL 做掉一半 | 128卡 | ★★ |
| 8 | VeriServe 原版：统一 reward serving | 极高 | ❌ 已死（DistRS, NSDI'26 正面占领） | — | — |

## 二、推荐的作战组合（三部曲）

**主线（容错叙事，你的比较优势）：**
1. **Idea A**（2026 内做完挂 arXiv，投 HPDC'27 tools&data 类目）——故障成本解剖 + 开源 trace + Calculon 式外推模型。与 RFT-FaultBench 用二维表切割（infra 故障 vs 语义故障 × 容错表征 vs 异常检测）。
2. **Idea B**（A 的数据做 motivation，6 个月内出稿防 Crab 组）——轨迹状态五元组（token×环境快照×权重版本×RNG×副作用 journal）的原子恢复协议；Crab/DeltaBox 降级为可插拔后端；备选动机"spot 抢占下的轨迹可移动性"（RLBoost 留的坑）。
3. **VeriServe 修正路线**（可作为 A 的延伸或 B 的一章）——"效率已被 DistRS 解决，正确性无人管"：区分模型真错 vs infra 假错（KAT-Coder：16% 轨迹含 sandbox 故障；DistRS 把 timeout 一律记负分正是污染源），置信标注 reward + deadline 预算内选择性重算。

**支线（通信库主场，可与师兄工作衔接）：**
- **RMcast**：先做 2-3 周 go/no-go 实验（8-16 卡测 checkpoint-engine/fabric-lib 的实际 injection bytes vs 手推下界，gap<20% 放弃，>50% 立项）。契约：ForestColl/TE-CCL/HeteCCL 必须做正面 baseline，证明 TE-CCL MILP 在数百 instance 时爆炸、ForestColl 下界在"源 sharded+多组 replica"不紧。payload 参数化吸收稀疏 delta 威胁。venue：SC/PPoPP。
- **RL-Calculon**：最快出手的一篇——8 月 PMBS@SC26 workshop 可先占坑（sync 长尾 order statistics + 配比公式 + verl 校准，4-6 页）。重定位为"simulator 给不出的 closed-form 决策规则 + 无人建模的 agentic 环节（tool-call 空等/KV 重算/长度分布漂移）"，RLSim/Frontier/Charon 转为对照 baseline。

## 三、8 个新 idea 的详细裁决

### Idea 1 RMcast（collective 原语）— medium
- **幸存空位**：1→N reshard-multicast 的闭式下界与结构刻画（shard-exchange × replica-relay 混合）、solver 爆炸规模下的多项式分解算法、churn 下增量 plan 修复+代价上界。
- **三大威胁**：TE-CCL(SIGCOMM'24) 归约威胁（"这只是一个 demand instance"）；Alpa(MLSys'23) 已做 1对1 mesh 下界；SparseRL-Sync 等稀疏 delta 传输（~100x 削减）攻击全量传输的前提。
- **事实纠错**：引用的 PCCL(2606.07019) 不存在；Meta NCCLX 已有 shrink/grow 动态 group——"首个 elastic collective"不能说。

### Idea 2 RL-Traffic Orchestrator — high
- 空位真实（≥4 类流量干扰 measurement + staleness budget 编译为网络调度语义，以训练质量为目标），但 DualPath 已做推理侧 QoS 隔离、RollArt 已刻画流量 QoS 差异。
- **事实纠错**：colocated 部署权重走 CUDA IPC 不过网——适用域必须钉死在"disaggregated 且共享 RoCE fabric"；NAIC'26/HotNets'26 均已截稿，最近窗口 NSDI'27 秋。

### Idea 3 RL-Calculon — medium
- **被证伪**："无公开预测工具"不成立——RLSim(OrchestrRL)、Frontier、Charon(MLSys'26 Oral) 三个 DES 已出，且 RLSim 出自 CUHK Hong Xu 组，6-12 个月内可能扩展成独立工具论文。
- **幸存空位**：closed-form O(1) 解析解（vs 分钟级仿真）+ agentic 特有环节建模 + 跨系统 blind holdout 校准 + 长度分布漂移参数化（回应 Libra 非平稳性攻击）。

### Idea 4 AgentRL-Sim — high
- "第一个 RL 闭环 DES"已被 RLSim 占。幸存三锚点：环境回路一等公民（tool 延迟 trace 回放/sandbox 超时/KV 驻留）、故障 what-if（RL 方向零竞品，绑定你的容错背景）、开放跨框架社区平台（现存全闭源）。eval 需多尺度验证阶梯 + 分布匹配指标（K-S/Kendall τ）。

### Idea 5 AuroraRL（超算 agentic RL）— medium
- 核心交集空置（无人在批处理超算端到端扩展 agentic RL），但 sandbox 支柱被 Polar(2605.24220) 侵蚀、弹性支柱被 BiDiRL(ATC'26) 部分占领。
- **硬前置**：≥512 卡 AMD/DCU 机时（国产 DCU 集群门槛最低；或 UIUC+ORNL 模式合作申 Frontier/LUMI）。拿不到就降级。主叙事应转为"第一个 leadership 级 AMD 超算上的 agentic RL scaling 研究"，把 Slingshot/RCCL 通信病理做第一贡献。

### Idea 6 SandFS（Lustre sandbox 存储）— medium
- 坑位真空（Rollout Infrastructure Tax 只测云 substrate，HPC/Lustre 留白），组内可行性最好：校级超算+8-64 卡即可。
- **必须重写 novelty**："metadata 当一等瓶颈"是伪命题（Shifter/Sarus/podman-hpc 十年前就为此而生）——真正差异是 agentic RL 的**每 step 成百上千次 create/reset/fork + 高 churn 可写状态**（metadata 写路径问题，SquashFS 只读方案不覆盖）。baseline 必须含 podman-hpc+GekkoFS/BeeOND 组合拳。风险：组内网络背景，FUSE/Lustre 学习曲线 2-3 个月。

### Idea 7 TrajStore — high
- **事实纠错**：TransferQueue 已有 NIXL/Ray RDT RDMA 后端且将成 verl v0.8 默认——"RDMA 绕开 Ray 序列化"的卖点一半已被做掉；Mooncake roadmap 明确写了 RL sample flow。
- 幸存机制：content-addressed 分段存储 + copy-on-write 谱系（GRPO 组内共享前缀/relabel 派生轨迹去重——Reverb/GEAR/TQ/Mooncake 都没有）。但需先做 1-2 个月数据面占比 measurement，占比 <10% 止损。

### Idea 8 VeriServe — 原版已死
- **DistRS**（PKU Xin Jin 组+字节 Seed，NSDI'26，2026-05 已宣讲）标题即"Disaggregated Reward Service for RLVR with Batch-Level Constraint"——group/batch-deadline 洞察、弹性伸缩、multi-tenant 全部做掉。
- 修正路线 1（推荐）：**可信 reward 服务**（见上文三部曲）；路线 2：group-SLO 感知的 judge 推理引擎内部机制（较弱，且 Stoica/Mirhoseini 团队可能在做）。

## 四、师兄三个 idea 的拥挤度结论（第一轮已交付，存档）

- **(1) KV cache**：通用命题 12+ 竞品（Seer/RollArt/DORA/Heddle/Libra/ROSE/WAR/MiniMax/GLM-5/Continuum）。真空白：训推边界 KV 复用（HF 博客点名零竞品）、权重更新后 KV delta 修复（ROSE 实测 >95% 稀疏）、KV 的 MVCC。
- **(2) tool call 重入调度**：红海——Heddle 正面做掉，Libra/WAR 已二轮内卷。错位切口：环境/沙箱与推理联合调度、colocated 架构下的重入。
- **(3) 权重同步**：三个里最拥挤。传输到硬件极限（fabric-lib 1T/1.3s），版本管理被 TensorHub/DORA/Laminar/StaleFlow 瓜分。
- **战略建议**：三合一为"长轨迹感知的训推协同运行时"（tool call 边界=版本切换点+KV 失效点）。

## 五、竞品速查表（写论文时的 related work 核心）

OSDI'26: Seer(2511.14617)/RollArt(2512.22560)/Weave/RobustRL(2512.22492)；EuroSys'26: Laminar；NSDI'26: RLBoost(2510.19225)/DistRS/RollPacker(2509.21009)/ForestColl/HeteCCL；ATC'26: BiDiRL(2607.09207)；MLSys'26: fabric-lib(2510.27656)/Charon；沙箱: Crab(2604.28138)/DeltaBox(2605.22781)/ACRFence(2603.20625)/Cordon(2606.17573)/Polar(2605.24220)；测量/benchmark: RFT-FaultBench(2605.04431)/RL in the Wild(2509.25279)/Rollout Infrastructure Tax(2607.01415)；权重: checkpoint-engine/AWex/TensorHub(2604.09107)/TENT/StaleFlow(2601.12784)；仿真: RLSim(2601.01209)/Frontier(2605.21312)；veRL=HybridFlow(EuroSys'25)。

## 六、venue 时间线（2026-07 后的可投窗口）

- **2026-08**: PMBS@SC26 workshop（RL-Calculon 占坑）
- **2026 内**: arXiv 占位（Idea A measurement 部分、SandFS measurement 段）
- **2027-01 左右**: HPDC'27（A 首选，tools&data 类目）、ICS'27
- **2027 春**: EuroSys'27 秋季轮（B）、SC'27（2027-04，RMcast/AuroraRL）
- **2027-06**: ACM ATC'27（兜底；注意 USENIX ATC 已停办）
- **2027-09**: NSDI'27 秋（流量调度类）

*详细验证原文：`E:\claude-data\tmp-ft-verdicts.txt`（容错 6 路）、`E:\claude-data\tmp-new-ideas.txt`（新 idea 8 份裁决）*
