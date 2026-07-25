#!/usr/bin/env bash
# Day 2 runner: concurrent NCCL groups on 112+113 (4 ranks).
#
# NCCL: PROTO=Simple ALGO=Ring GID=3 SOCKET=bond0; per-GPU HCA pin.
# Python: Megatron_test conda (torch 2.11 + NCCL 2.28.9)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/bench"
PY="${PYTHON:-/home/xiajinyi25/Megatron_test/conda_env/bin/python}"
MPIRUN="${MPIRUN:-/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpirun}"
HOST_TRAIN="${HOST_TRAIN:-192.168.5.112}"
HOST_INFER="${HOST_INFER:-192.168.5.113}"
OUTDIR="${OUTDIR:-$EXP/results/concurrent_$(date +%Y%m%d_%H%M%S)}"
# NOTE: do not name this GROUPS — bash has a readonly GROUPS array.
GROUP_LIST="${GROUP_LIST:-1,2,4,8,16}"
CHANNEL_LIST="${CHANNEL_LIST:-}"   # e.g. 4,8,16
ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-2}"
# Use 4 GiB payload to leave headroom for NCCL workspace on 40GB A100
W_BYTES="${W_BYTES:-4294967296}"
CONDA_LIB="$(dirname "$PY")/../lib"

mkdir -p "$OUTDIR"
tar cf - -C "$ROOT" bench/concurrent_groups.py experiments/go_nogo/ib_counters.py | \
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

echo "OUTDIR=$OUTDIR GROUP_LIST=$GROUP_LIST CHANNEL_LIST=$CHANNEL_LIST W_BYTES=$W_BYTES"
set +e
"$MPIRUN" --allow-run-as-root --hostfile "$OUTDIR/hostfile" -np 4 --map-by slot \
  -x NCCL_PROTO -x NCCL_ALGO -x NCCL_IB_GID_INDEX \
  -x NCCL_SOCKET_IFNAME -x NCCL_NET_GDR_LEVEL -x NCCL_DEBUG \
  -x GLOO_SOCKET_IFNAME \
  bash -c '
    export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
    export RANK=${OMPI_COMM_WORLD_RANK:-0}
    export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
    export MASTER_ADDR='"${HOST_TRAIN}"'
    export MASTER_PORT='"${MASTER_PORT:-29571}"'
    export LD_LIBRARY_PATH="'"$CONDA_LIB"':${LD_LIBRARY_PATH:-}"
    if [[ "$LOCAL_RANK" == "0" ]]; then export NCCL_IB_HCA=mlx5_1; else export NCCL_IB_HCA=mlx5_3; fi
    exec "'"$PY"'" -u "'"$EXP"'/concurrent_groups.py" \
      --outdir "'"$OUTDIR"'" --groups "'"$GROUP_LIST"'" --iters '"$ITERS"' --warmup '"$WARMUP"' \
      --channels "'"$CHANNEL_LIST"'" --w-bytes "'"$W_BYTES"'"
  ' 2>&1 | tee "$OUTDIR/run.log"
rc=${PIPESTATUS[0]}
echo "mpirun_rc=$rc" | tee -a "$OUTDIR/run.log"
exit "$rc"
