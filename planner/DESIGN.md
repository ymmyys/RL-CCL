# RMcast Planner — 设计文档 v0.1

*2026-07-24。基于四模式实验(run_20260724_132308) + ForestColl/TACCL/TE-CCL 精读 + 对抗验证。*

---

## 0. Novelty 定位（P0 修正，直接决定论文怎么写）

**"我们定义了一个新 collective 语义"这个 claim 已死，不许再写。**
TE-CCL (SIGCOMM'24) 的 demand 张量 D[s,d,c]∈{0,1} + copy 变量可以一字不差表达
RMcast（sharded 源→N 个重排副本，甚至 churn 事件=新 demand 矩阵）。审稿人一句
"why not feed it to TE-CCL"就能击穿。

**正确定位**："RMcast 是 TE-CCL 可表达 demand 族的一个**结构化子族**；我们证明该
子族在 rail-optimized 拓扑上存在**近线性、可证最优的规划算法** + **O(churn 规模)
的增量修复** + **可直接落 NCCL/RDMA 的执行输出**，而通用 solver 需要小时级求解、
数百 GB 内存、且完全离线。"

定量弹药（已核实的先验数字）：
- TE-CCL：128 GPU AllGather 1300s–7h，256 GPU 2.8h，~350GB 内存，2h timeout；
- TACCL：128 GPU ~11h，>4 chassis 8h 无解/OOM(>400GB)，跨节点链路人工预选；
- 而我们的传输本身只要 **1.68s**，RL 推理实例分钟级增删——**solver 时间 >> 传输
  时间**是通用 solver 在这个场景的死穴。
- 必须**自己复现** TE-CCL 数字（不能只引论文），并在小规模上拿它当 optimality
  oracle 反向加分（Day 3 任务）。
- Blink (MLSys'20) 是"结构换多项式"先例（单机 NVLink 生成树打包），必须引用划界。

**幸存的 delta（对抗验证后）**：
1. **复杂度/在线性**：结构化多项式 vs 通用 MILP（定量可打）；
2. **Steiner 目标子集**：ForestColl 的 spanning tree packing 靠 min-cut=packing
   对偶拿多项式最优，目标子集≠全体节点时变 directed Steiner packing，对偶断掉
   （一般 NP-hard）——这是数学上过硬的断点；
3. **demand-aware 下界**：ForestColl 式★对 RMcast 不紧（详见 §2）；
4. **多 owner 择根**（Megatron DP>1 时同一 shard 多个持有者）；
5. **churn 增量修复**：所有先验都是离线全量重算（ForestColl 1024 GPU 级 36 分钟）。

**被降级的**：非平凡 reshard（TP2→TP4）不是网络层 novelty——引理 0 自己证明了
addr 不影响链路字节量；当每实例落在单主机内时 reshard 完全塌缩为机内寻址。
reshard 是**执行层贡献**（需求类构造 + 机内扇出的寻址消解）。

---

## 1. 问题形式化

**记号。** 权重地址空间 𝒲=[0,W)。源 GPU 集合 S，s∈S 持有 σ(s)⊆𝒲（区间并），
∪σ(s)=𝒲，允许重叠（TP 复制的 norm 参数）。目标：N 个实例，实例 j 的 GPU d∈D_j
有需求 τ(d)⊆𝒲 与寻址函数 addr_d；∪_{d∈D_j} τ(d)=𝒲。拓扑 G=(V,E)，V=GPU∪NIC∪
交换机，边带宽 b_e，区分机内/网络链路。

**引理 0（重排=端点寻址）。** 传输期间字节以逻辑地址标识，addr_d 只在落盘时应用，
不影响链路字节量。传输问题只依赖需求关系 R={(d,b): b∈τ(d)}。布局对齐时问题退化
为 |S| 个独立整 shard multicast。

**需求类。** 𝒲 按 (holders(b), dst(b)) 分等价类 = 所有区间端点 overlay 出的原子
区间，K ≤ 2M（M=输入区间端点总数）——**K 关于输入规模多项式**，算法可行性的基础。
⚠️ 实际 K 可上千（7B 模型按张量边界 overlay）——见 §5 需求类合并。

**输出。** 传输 plan：带时序的 send/copy DAG，满足因果性（store-and-forward，
chunk→0 得流体模型）、容量、完备性。**目标：**最小化 makespan T。

---

## 2. 下界（含对抗验证修正）

**统一割界。** 对任意 A⊆V：
X(A) = {b : **holders(b)⊆A** ∧ dst(b)⊄A}，cap(A) = Σ_{e∈δ⁺(A)} b_e，
**T ≥ max_A |X(A)|/cap(A) =: LB_cut。**

> ⚠️ **修正 1（原稿错误）**：X(A) 必须用 holders(b)⊆A（所有初始持有者都在 A 内），
> 不能用钦定的 owner(b)∈A——否则 DP>1 时另一副本在 A 外，字节根本不必穿割，
> 钦定 owner 版只是"从钦定源发货"这类 plan 的下界，不是任意 plan 的下界。
> 若第一篇论文 scope 到 DP=1，须显式声明。

> ⚠️ **修正 2（可达性）**：LB_cut 在一般有向图上**可能无 plan 达到**（directed
> Steiner packing 无 min-cut=packing 对偶；网络编码文献中有向图 gap 可达
> Ω(log n)）。全文只声明 LB_cut 为下界；可达性只在 rail-optimized Clos 类上
> 构造性证明。流体/chunk 化逼近关系照 ForestColl Appendix D 的方式声明。

**三个典范割**：(a) 源聚合 egress 割 → 源侧总发送量下界 = W（dataflow-optimal
的定义）；(b) 单链路割；(c) 每副本 ingress 割 ≥ W/ingress_bw。

**三紧最优性定义。** plan 在瓶颈割 A* 上：**量紧**（每字节恰穿一次、无多余）、
**衡紧**（各边负载∝带宽）、**时紧**（割持续饱和）。流体模型下三紧 ⟺ T=LB_cut
（需补多 argmax 割情形：T=LB 则在每个 argmax 割上三紧）。

**四模式作为紧性的证伪矩阵**（这是实验与理论的正确对接方式）：

| 模式 | 量紧 | 衡紧 | 时紧 | 实测 |
|---|---|---|---|---|
| A 串行广播 | ✓(正向) ✗(回流 W/2) | ✗ 单 NIC | — | 2.37s |
| B 直连 P2P | ✗ 穿割 2W | ✓ | — | 3.47s |
| C 顺序 relay | ✓ | ✓ | ✗ 两阶段串行 | 1.94s |
| D 并行广播 | ✓ | ✓ | ✓(近似) | 1.68s |

> ⚠️ **修正 3（实证-理论对齐，审稿人两分钟能算出来的矛盾）**：双 NIC 线速割界
> ≈0.61s（goodput 口径≈0.69s），模式 D 实测 1.68s 是下界的 **2.4×**——不许写
> "D 达到 LB"。必须：测 NCCL broadcast busbw 基线，全部改用"占可达割容量的 X%"
> 报告，解释 D 的常数损耗（channel 数/流水建立/H2D）；或把四模式验证诚实降级为
> "每模式恰证伪一条紧性、排序吻合"。A/D 比值实测 1.41×（非 2×）也说明回流与
> 前向有重叠——序数结论稳，倍数结论不稳。

---

## 3. 结构猜想：最优解的规范形

层级拓扑（rail-optimized 两层 Clos + 机内互联）上存在 dataflow-optimal plan：
每需求类 a：**跨机一次**（owner 主机→每需求主机的 leader，按 rail 流水链/树）
→ **机内扇出**（leader 经 NVLink/PCIe 分发本机需求 GPU，reshard 复杂性在此由
addr 消解）→ **NIC 间 waterfilling**（需求类到 (源 NIC,目标 NIC,rail) 按带宽
比例均衡）。

不损最优的条件：(i) 满 bisection ⇒ 瓶颈割只有源 egress/目标 ingress/单 NIC 三类
——**⚠️ 这是断言不是定理，必须升级为 hose-model 可路由性引理并证明**（满 bisection
下任何满足 per-NIC 速率约束的流量矩阵可路由）；(ii) 机内带宽 ≥ (r_h−1)×b_NIC；
(iii) 需求类可分数拆分。
⚠️ rail-optimized **不是** rail 间任意连通——跨 rail 走 spine 或 PXN，owner 的
shard 分布与 rail 不匹配时均衡从 waterfilling 升级为带 rail 约束的运输 LP
（仍多项式，闭式没了，条件 (ii) 要重推）。

已知损失情形：异构 NIC（整数粒度 LPT 界）、机内贫弱（PCIe，gap 由 b_NIC/b_intra
定）、过订阅核心层（需跨类联合路由，流体下仍 LP 可解）、colocation（未证，open）。

---

## 4. Planner 算法骨架（六步，近线性-低次多项式）

1. **需求类构造** O(M log M)：区间端点 overlay → 原子区间 → 按 (holders, 目标主机集) 合并。
2. **主机级收缩** O(K·H)：GPU 收缩到主机；机内需求直接生成 copy，移出网络规划。
3. **rail/NIC 均衡** O(K·H·R) 或小 LP：waterfilling / 带 rail 约束的运输 LP；整数版 LPT。
4. **每类跨机结构** O(K·H log H)：单目标→点对点；多目标→按 rail 流水链（大 H 时
   与 log 深树做 α-β 相变切换，参数化）。
5. **机内扇出**：leader→本机需求 GPU，与网络接收 chunk 级流水。
6. **chunk 化 + DAG 生成**：每需求类映射为一个 NCCL 组（broadcast）或 RDMA WRITE 序列。

**内建 sanity check**（单元测试）：喂 2+2 实验拓扑 → 必须输出模式 D 的组划分；
喂单 NIC 拓扑 → 必须输出模式 A 的链结构但**无回流**（demand 驱动 DAG 自然消除
全局组的语义冗余）。

**⚠️ 系统级 go/no-go（对抗验证列为最高系统风险）**：K 上千时"每需求类一个 NCCL
组"不可行（communicator 建立开销、SM/channel 占用、并发组 QP 竞争）。必须加
**需求类合并步**（合并同 (源 NIC,目标主机集) 的类，组数压到 O(R·H)），并先跑
并发组 microbenchmark（Day 2）——若 8 组并发聚合 goodput < 单组 60%，时紧系统性
失效，合并从优化项升级为设计核心。

---

## 5. Churn 增量修复

范围：**迭代间 churn**（版本 k 传完后实例增删；传输中途 abort 是 future work）。
局部性来源：规范形 = 每需求类独立结构 + NIC 负载表。实例 j 增删只触碰 j 的区间
端点相邻的原子（O(M_j log M)）、每棵结构上 j 主机一个节点、负载表一行。
- **join**：主机已在 H(a) → 纯机内加扇出项，网络 plan 零改动；新主机 → 按 rail
  插入链尾。编辑量 O(K_j)。
- **leave**：链上 splice，O(1)/类；规范形中 relay 只服务本机 GPU，不跨主机中转
  他人 → 退出不孤儿化任何第三方（结构红利，写进论文）。
- 对比 NCCL abort-rebuild：全局屏障 + 全量重建 O(N) vs 编辑 O(K_j) 与集群规模
  解耦——**这是可严证的命题**；(1+ε) makespan 界在"新主机需求集中单 rail"时论证
  不成立，降级为实验对照，或补证。

---

## 6. 先验对照矩阵（related work 骨架）

| 系统 | demand 模型 | 复杂度 | 最优性 | churn | 执行输出 |
|---|---|---|---|---|---|
| ForestColl (NSDI'26) | 全员互收(allgather 族)，§5.7 仅根量非均匀 | 多项式(1024GPU~36min) | 任意拓扑吞吐最优(spanning tree 对偶) | 无 | schedule |
| TACCL (NSDI'23) | 任意(chunk pre/post condition) | MILP,128GPU~11h,更大 OOM | 无保证 | 无 | MSCCL |
| TE-CCL (SIGCOMM'24) | **任意 0/1 张量(能表达 RMcast)** | MILP/A*,128GPU 1300s-7h,350GB | 小规模最优/A* 次优 | 无 | MSCCL |
| Blink (MLSys'20) | broadcast/allreduce | 多项式(树打包) | 单机 NVLink | GPU 子集动态 | 自有运行时 |
| checkpoint-engine | broadcast(+内建一级 relay) | — | 无 | 无 | NCCL/mooncake |
| fabric-lib | 直连 P2P(N×W) | 静态 schedule | 无 | 无 | RDMA WRITE |
| **RMcast(我们)** | 结构化子族(sharded 源→子集多播) | **近线性** | **限定拓扑类可证** | **O(K_j) 增量** | **NCCL 组/RDMA** |

ForestColl 可复用资产：auxiliary network+maxflow+二分求瓶颈割（改造成 RMcast
下界计算器，demand 边按需求字节设容量，多 owner 用虚拟超根表达）；edge splitting
消 switch（与 demand 无关，整体搬用）；tree-flow 流水执行模型（模式 D 胜出的理论
解释）。断裂处：Steiner packing 无对偶、跨 shard 联合带宽分配、多 owner、增量性。

还需补查：veRL/HybridFlow（reshard 语义出处）、MSCCL++、近三月 RL 权重同步新论文
（方向正热，月度扫描）。

---

## 7. 开放问题（按风险排序）

1. 一般图 NP-hard（broadcast time 问题）——引用并把"多项式+最优"限定在层级拓扑类；
2. 多组并发时紧假设——Day 2 实验是 go/no-go；
3. 过订阅/异构下分解 gap 是否有界——查分数 Steiner 打包整数 gap 文献；
4. α-β 延迟项：链 O(H) vs 树 O(log H) 相变，Step 4 参数化；
5. colocation（训推同宿主）下分解最优性未证；
6. 传输中途 churn + 权重版本一致性——诚实标 future work（与 Idea B 轨迹一致性有接口）。

---

## 8. 第一周任务清单（服务器执行）

| Day | 任务 | 产出 | go/no-go |
|---|---|---|---|
| 1 | ib_write_bw + nccl-tests broadcast busbw 基线；四模式改按"占割容量%"报告 | `bench/nic_baseline.sh`, `analysis/lb_table.md` | D 效率是否 >80%（对 NCCL busbw 口径） |
| 2 | **并发组实验**：W 切 2/4/8/16 shard 各一 NCCL 组并发，测聚合 goodput 衰减；扫 NCCL_MAX_NCHANNELS/chunk | `bench/concurrent_groups.py` + 曲线 | 8 组衰减是否 <20%；红则合并步升级为设计核心 |
| 3 | TE-CCL oracle：克隆 artifact，把 2+2 拓扑+RMcast demand 编码求精确 MILP，记目标值与求解时间 | `experiments/teccl_oracle/` + results.csv | solver 时间 vs 1.68s；验证 D 是否小规模最优 |
| 4 | 重写下界：引理 1(多 holder 割界+证明)、命题 2(三紧+多 argmax)、可达性 caveat；DP=2 toy example 检验 | `theory/bounds.tex` 2-3 页 | — |
| 5 | hose 引理攻坚：精确定义拓扑类，证明或找 TE 文献现成结果；列跨 rail/PXN 反例候选 | `theory/topology_lemma.tex` | 证不动则最优性范围收窄 |
| 6 | planner 原型 v0：Step 1-3 (~300 行 Python)+两个 sanity 单元测试 | `planner/rmcast_plan.py` + tests | 测试须通过 |
| 7 | 先验矩阵成文 + arXiv 近三月扫描 + 周报(三个 go/no-go 裁决+下周分支) | `related_work/matrix.md` | 全绿→4 节点/TP2→TP4 不对齐实验；并发红→合并设计 |

所有实验固定 NCCL 版本与环境变量，写进脚本头注释（复现性）。Day 1-3 相互独立可乱序。
