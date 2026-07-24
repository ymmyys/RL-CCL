# Cluster env checklist (112 / 113) — fabric-lib & go/no-go

| Requirement | Spec | 112/113 status |
|---|---|---|
| RDMA | libibverbs, ≥1 NIC/GPU | ✅ mlx5_0..3, RoCEv2; active `mlx5_1`/`mlx5_3` @100Gb |
| NIC | CX-7 / EFA preferred | ✅ BlueField-3 integrated **ConnectX-7** (MT43244) |
| GPUDirect / dmabuf | GPU dmabuf | ✅ kernel dmabuf present; Warden relay uses it |
| Kernel | Linux ≥5.12 | ✅ `5.15.0-181-generic` |
| CUDA | ≥12.8 | ✅ driver UMD 13.3; toolkit `/usr/local/cuda` 13.3; torch `2.11+cu128` |
| Nodes | 2–4 | ✅ `192.168.5.112` (xfusion-2), `192.168.5.113` (xfusion-3); 2×A100-40GB each |
| GDRCopy | needed by fabric-lib | ❌ not installed |
| libfabric | needed by fabric-lib | ❌ not installed (`libfabric-dev` in apt; needs sudo) |
| Docker | pplx-garden-dev image | ❌ no docker binary |
| SYS_PTRACE/ADMIN | pidfd_getfd | ⚠️ no passwordless sudo; try user ns later |
| Megatron | reference stack | ✅ `/home/xiajinyi25/Megatron_test` (+ conda_env on both nodes) |
| NCCL knobs | RoCE | `NCCL_IB_HCA=mlx5_1 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_GID_INDEX=3` |

RoCE data IPs (ens7f1np1 / mlx5_1):
- 112: `10.99.3.2`
- 113: `10.99.3.3`

Homes are **not** NFS-shared — `run_go_nogo.sh` tar-pipes experiment code to 113.
