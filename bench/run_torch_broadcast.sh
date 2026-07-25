#!/usr/bin/env bash
# Torch/NCCL broadcast baseline (same stack as reshard_real).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/bench"
PY="${PYTHON:-/home/xiajinyi25/Megatron_test/conda_env/bin/python}"
MPIRUN="${MPIRUN:-/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpirun}"
HOST_TRAIN="${HOST_TRAIN:-192.168.5.112}"
HOST_INFER="${HOST_INFER:-192.168.5.113}"
OUTDIR="${OUTDIR:-$EXP/results/torch_bcast_$(date +%Y%m%d_%H%M%S)}"
CONDA_LIB="$(dirname "$PY")/../lib"

mkdir -p "$OUTDIR"
tar cf - -C "$ROOT" bench/torch_broadcast_baseline.py | \
  ssh -o BatchMode=yes "$HOST_INFER" "mkdir -p '$ROOT' && tar xf - -C '$ROOT'"

cat > "$OUTDIR/hostfile" <<EOF
${HOST_TRAIN} slots=2
${HOST_INFER} slots=2
EOF

export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME=bond0

"$MPIRUN" --allow-run-as-root --hostfile "$OUTDIR/hostfile" -np 4 --map-by slot \
  -x NCCL_PROTO -x NCCL_ALGO -x NCCL_IB_GID_INDEX \
  -x NCCL_SOCKET_IFNAME -x NCCL_NET_GDR_LEVEL -x NCCL_DEBUG -x GLOO_SOCKET_IFNAME \
  bash -c '
    export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
    export RANK=${OMPI_COMM_WORLD_RANK:-0}
    export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
    export MASTER_ADDR='"${HOST_TRAIN}"'
    export MASTER_PORT='"${MASTER_PORT:-29572}"'
    export LD_LIBRARY_PATH="'"$CONDA_LIB"':${LD_LIBRARY_PATH:-}"
    if [[ "$LOCAL_RANK" == "0" ]]; then export NCCL_IB_HCA=mlx5_1; else export NCCL_IB_HCA=mlx5_3; fi
    exec "'"$PY"'" -u "'"$EXP"'/torch_broadcast_baseline.py" --outdir "'"$OUTDIR"'"
  ' 2>&1 | tee "$OUTDIR/run.log"
