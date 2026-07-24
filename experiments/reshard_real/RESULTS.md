# Real Megatron→vLLM reshard — RESULTS

**Run:** `results/run_20260724_132308/` (canonical; dual-NIC via per-GPU `NCCL_IB_HCA`)  
**W** = 15,231,233,024 B ≈ **14.19 GiB** (Qwen2.5-7B BF16; matrix asserted exact)  
**Topo:** 113 GPU0↔GPU1 = **SYS** (no NVLink); PCIe copy ≈14.6–22 GB/s  
**Binding:** LOCAL_RANK0→`mlx5_1`, LOCAL_RANK1→`mlx5_3` (Active RoCE).  
Note: `NCCL_IB_HCA="=mlx5_1,=mlx5_3"` is broken on this NCCL — silently drops mlx5_3.

## Four-mode table (5 iters, interleaved; mean±std)

| Mode | Semantics | 112 egress | mlx5_1 / mlx5_3 xmit | Wall time | vs §6 expect |
|---|---|---:|---:|---:|---|
| **A** serial per-shard broadcast | CE-style owner broadcast, global PG | **1.059×W** | 0.01 / **16.12** GB (single rail) | **2.37±0.23 s** | egress ✓; time >1.4s (OK, Simple+ring) |
| **B** direct P2P | fabric-lib-style; each shard → both infer ranks | **2.118×W** | **16.13 / 16.13** GB | **3.47±0.11 s** | egress ✓; time high (eager P2P serialized — see warning) |
| **C** relay | 0→2 & 1→3 then 2↔3 swap | **1.059×W** | **8.06 / 8.06** GB | **1.94±0.11 s** | egress+dual-NIC ✓; PCIe phase2 ~0.95s |
| **D** parallel shard broadcast | pg{0,2,3}+pg{1,2,3} concurrent | **1.059×W** | **8.07 / 8.07** GB | **1.68±0.30 s** | egress+dual-NIC ✓; fastest |

Mode C phases (max across ranks): p1≈1.23s (wire), p2≈0.95s (113 PCIe swap); sum≈2.18s, overlap LB≈1.23s.

## Mode A ring evidence (auto-relay)

From NCCL logs (`NCCL_ALGO=Ring`):

- rank1: `Ring 00 : 0 -> 1 -> 2`
- rank2: `Ring 00 : 1 -> 2 -> 3`
- rank3: `Ring 00 : 2 -> 3 -> 0`

113 counters in A: **mlx5_1 rcv≈16.12G** and **mlx5_3 xmit≈8.06G** — ingress on one NIC, forward egress on the other. Matches the prediction that NCCL ring already relays on this 2+2 topology; trainer egress stays ≈W without an explicit relay schedule.

GDRDMA present in channel setup (`via NET/IB/0/GDRDMA`).

## Conclusions (for paper motivation)

1. **Bytes:** Deployed-style **P2P (B)** injects **≈2.1×W**; broadcast/relay-shaped paths (**A/C/D**) stay **≈1.06×W** (header tax). Gap ≫50% on the byte axis → go signal for studying better dataflows unchanged.
2. **Dual-NIC:** Explicit **C** and reviewer baseline **D** both split ≈W/2 per Active NIC; **A** collapses forward progress onto one rail. So “egress≈W” alone is not enough — **A does not get multi-NIC parallelism**.
3. **C ≈ D** on this N=2 setup (D slightly faster). Novelty is **not** “invent relay for N=2”; it is scheduling under general N / multi-node / nontrivial reshard / churn, while today’s stacks miss the conjunction of egress≈W **and** multi-NIC (A) or pay N×W (B).
4. **Hardware caveat:** 113 is **PCIe/SYS only**. C’s phase-2 swap is ~0.95s; “2× faster” claims must be conditional on NVLink (not present here).
5. **Related-work honesty:** checkpoint-engine P2P already has a one-hop relay (trainer egress≈W). Do not claim “nobody relays”; claim the four-way conjunction in INSTRUCTIONS §8.

## Artifacts
- `gen_traffic_matrix.py`, `reshard_bench.py`, `run_reshard.sh`, `ENV.md`
- `results/run_20260724_132308/{summary.json,nccl_rank*.log,run.log,traffic_matrix.json}`
- Earlier single-NIC misconfig runs kept under `run_20260724_131902` / `132116` for forensics
