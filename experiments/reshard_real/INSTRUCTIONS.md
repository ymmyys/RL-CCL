# 真实 Megatron→vLLM 权重传输实验 — 执行说明

> 给服务器上的 Claude 会话：按本文件实现并运行实验。前置事实已在本机侧核实
> （Megatron TP 切分规则、Qwen2.5-7B 架构常数、checkpoint-engine/fabric-lib 真实数据路径、
> 方法学对抗审查），照做即可，不需要重新调研。

## 0. 实验目的（一句话）

把 go/no-go 的合成 2GiB 换成 **Qwen2.5-7B 的真实 reshard 流量矩阵**，在 112→113 上
对比四种权重分发模式的 **网卡出口字节数** 和 **wall time**，为 RMcast 论文的
motivation 章节产出第一张真实数据图。

## 1. 前置检查（必须先做，结果写进 ENV.md）

```bash
# 1a. 两台机器的 GPU 互联拓扑 —— relay 模式的时间结论押在这上面
nvidia-smi topo -m          # 112 和 113 都要跑，记录 GPU0-GPU1 之间是 NV# 还是 PIX/PHB/SYS

# 1b. GPUDirect RDMA 可用（之前已确认 nvidia_peermem 已加载，跑 NCCL 时在日志里再验证出现 GDRDMA）
lsmod | grep peermem

# 1c. 跨 rail 可达性：p2p 模式需要 112的mlx5_1 ↔ 113的mlx5_3 互通
# 用 ib_write_bw 测一次跨 rail 单向带宽（server 在 113 的 mlx5_3，client 在 112 的 mlx5_1）
# 如果跨 rail 不通或明显降速，记录下来——p2p 模式的流量路径要按实际情况解释
```

若 113 是 PCIe-only（无 NVLink），额外跑 CUDA samples 的 p2pBandwidthLatencyTest
或用 torch 测一次 GPU0↔GPU1 拷贝带宽，记录数字。

## 2. 流量矩阵生成（gen_traffic_matrix.py）

**不载入真实权重**——按 Qwen2.5-7B 架构常数生成真实 shape 的随机 BF16 张量。
以下常数已从 HF 官方 config.json 核实，直接用：

| 常数 | 值 |
|---|---|
| hidden_size | 3584 |
| num_hidden_layers | 28 |
| num_attention_heads | 28 |
| num_key_value_heads | 4 (GQA) |
| head_dim | 128 |
| intermediate_size | 18944 |
| vocab_size | 152064 |
| tie_word_embeddings | **false**（lm_head 独立，别漏） |
| qkv bias | 有（Qwen2 特有；o_proj 和 MLP 无 bias） |
| **W (BF16 全量)** | **15,231,233,024 B ≈ 14.19 GiB**（7,615,616,512 参数） |

**Megatron TP=2 切分规则**（源 rank0/rank1 各持有的部分）：

- qkv（融合 ColumnParallel，沿输出维切）：GQA 按 query group 分，每 rank 14 个 q 头 + 2 个 k 头 + 2 个 v 头 → 每 rank 每层 [2304, 3584]；qkv bias 同切（2304/rank）
- o_proj（RowParallel，沿输入维切）：每 rank 每层 [3584, 1792]
- gate/up（融合 ColumnParallel）：每 rank 每层 2×[9472, 3584]；down（RowParallel）：[3584, 9472]
- embed_tokens 和 lm_head（vocab-parallel）：各每 rank [76032, 3584]，两者独立
- RMSNorm（每层 2 个 + final）：**复制不切分**，字节记账时只算一份（由 rank0 发）

校验：每 rank shard ≈ 7,615,820,800 B ≈ 7.09 GiB；两 rank 之和 − 重复的 norm ≈ W。
脚本要打印逐张量清单和总字节数，与上面数字对账，误差为 0 才继续。

**易错点**（已知的坑，写代码时对照）：ColumnParallel 切 dim0、RowParallel 切 dim1
（与字面直觉相反）；k/v_proj 是 512×3584 不是 3584×3584；lm_head 别漏别重；
bias 只有 qkv 有；norm 不除以 TP。

## 3. 四种模式（reshard_bench.py，torch.distributed + NCCL，4 ranks）

rank0/1 = 112 的 GPU0/GPU1（源，Megatron TP=2）；rank2/3 = 113 的 GPU0/GPU1
（目标，2 个 TP=1 vLLM 实例，各需全量 W）。张量按 (src→dst) 打包成 ~256MB bucket。

| 模式 | 语义 | 对应真实系统 | 预期 112 egress |
|---|---|---|---|
| A. per-shard broadcast（串行） | 全局 communicator{0,1,2,3}，逐 bucket 由 owner rank 做 dist.broadcast（rank0 的 bucket src=0，rank1 的 src=1），串行 | checkpoint-engine broadcast 模式（bucket owner 广播，无跨网 gather——**不要**加 gather-to-rank0 步骤） | ≈1.0×W，但跨网前向流量集中在 ring 的单条跨节点边（单 NIC） |
| B. direct P2P | rank0 和 rank1 各自用 batch_isend_irecv 把自己的 shard 发给 rank2 **和** rank3 | fabric-lib（每目标实例独立收全量，无目标间转发） | ≈2.0×W，双 NIC 各承载 W |
| C. relay（我们的提案） | rank0→rank2、rank1→rank3 各发自己的 shard（egress=W，双 NIC 各 W/2）；然后 rank2↔rank3 在 113 节点内用 NCCL send/recv 互换缺失的一半 | RMcast 的 replica-relay | ≈1.0×W，双 NIC 各 W/2 |
| D. 并行分 shard broadcast（**必须加**） | 两个 communicator {0,2,3}(root 0) 和 {1,2,3}(root 1) 并发 broadcast 各自 shard | 审稿人反例基线：纯 NCCL 就能达到的 relay 等价物 | ≈1.0×W，双 NIC 各 W/2 |

模式 C 的节点内互换：优先实现 bucket 级流水线（边收边转发）；如果复杂就先做
顺序版（收完再换），但要**分别计时两阶段**并同时报告 sum 和 max（max = 完美
overlap 的下界）。

## 4. NCCL 环境钉死（否则字节数对不上账）

```bash
export NCCL_PROTO=Simple        # LL 协议会使线上字节 ~2x，LL128 +6%
export NCCL_ALGO=Ring           # Tree 会改变中间节点 egress
export NCCL_IB_HCA="=mlx5_1,=mlx5_3"   # 注意 = 前缀是精确匹配
export NCCL_IB_GID_INDEX=3      # RoCEv2
export NCCL_SOCKET_IFNAME=<管理网口名>  # 钉住 bootstrap，防止走错网
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH    # 日志留档：ring 顺序是重要证据（见 §6）
```

## 5. 测量协议

1. 每模式独立 communicator，**在计数器窗口外创建**；2-3 次 warmup 后再开始计量。
2. 字节：读 112 **和** 113 两台的 `/sys/class/infiniband/{mlx5_1,mlx5_3}/ports/1/counters/port_xmit_data`
   前后差值 ×4（单位是 4 字节 lane word）。读 113 的目的：验证 relay 的节点内交换
   确实不出网卡，并捕捉模式 A 中 ring 的 3→0 回流边。
   计数器含 ~3-6% 包头/ACK 开销，1.03-1.06×W 是正常读数，不要当成低效。
3. 时间：所有 rank `torch.cuda.synchronize()` + gloo barrier（别用 NCCL barrier
   污染 IB 计数器），取跨 rank 的 max。
4. 每模式 5 次 iter，四种模式**交错跑**而非连续跑（排除热漂移），报均值±标准差。
5. 顺手查 `hw_counters` 下的 out_of_sequence / np_cnp_sent，确认无重传/拥塞污染。

## 6. 预期结果表（用于对账，偏差大就先查环境再怀疑理论）

W=14.19 GiB，单 NIC 有效 ~92Gbps（≈11.5GB/s，fabric-debug 实测）：

| 模式 | 112 egress | NIC 利用 | 预计耗时 |
|---|---|---|---|
| A 串行 broadcast | 1.03-1.06×W | 跨网前向集中单边 | ≈1.4s |
| B direct P2P | 2.06-2.12×W | 双 NIC 各 W | ≈1.4-1.5s |
| C relay | 1.03-1.06×W | 双 NIC 各 W/2 | NVLink: 0.7-0.9s；PCIe-only: 1.1-1.5s |
| D 并行 broadcast | 1.03-1.06×W | 双 NIC 各 W/2 | ≈C |

**重要预警**：模式 A 的 NCCL ring 大概率自动形成"跨网一跳 + 节点内接力"，
所以它的 egress 本来就 ≈W——这不是 bug，恰恰是论证素材：NCCL 在这个退化
拓扑下自己就会 relay。保存 NCCL_DEBUG 日志里的 ring 顺序作为证据。
模式 C 和 D 结果接近也是**预期中的**（D 就是 C 的纯 NCCL 实现）——论文的
novelty 不在 N=2 的 relay 本身，而在一般 N、多目标节点、非平凡 reshard 下的
调度算法；这个实验只负责证明"已部署系统（A 的单边串行 / B 的 N×W）都没有
同时拿到 egress≈W 和多 NIC 并行，而最优 dataflow 在小规模上可达"。

## 7. 产出与提交

- `experiments/reshard_real/{gen_traffic_matrix.py, reshard_bench.py, run_reshard.sh}`
- `experiments/reshard_real/results/<时间戳>/`：每模式的计数器 JSON、时间、
  NCCL 日志（至少留 ring 拓扑部分）
- `experiments/reshard_real/ENV.md`：topo -m 输出、跨 rail 测试结果、节点内带宽
- `experiments/reshard_real/RESULTS.md`：结果表 + 与 §6 预期的对照 + 结论一段
- commit 到 main（本机会中转 push 到 GitHub）

## 8. 论文表述备忘（写 RESULTS.md 结论时对照）

- checkpoint-engine 的 P2P 模式**已内建一级 relay**（RDMA READ 拉取 + 子组内
  broadcast，trainer egress≈W），不要在论文里写"没有系统做 relay"；幸存的
  空白是"同时做到 egress≈W + 多 NIC 并行 + layout 感知 reshard + churn 适应"。
- fabric-lib 线上传的是 FP8（W 减半）且 N×W 摊在 256 张训练 NIC 上——聚合
  字节数忠实，单 NIC 时间不可直接外推，只 claim 字节维度。
- 所有"快 2×"的表述必须限定硬件条件（113 的互联类型）。
