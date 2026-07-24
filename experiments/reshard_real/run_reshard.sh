#!/usr/bin/env bash
# 4-rank Megatron→vLLM reshard bench: 112 (2 GPU) + 113 (2 GPU)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/home/xiajinyi25/Megatron_test/conda_env/bin/python}"
MPIRUN="${MPIRUN:-/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpirun}"
HOST_TRAIN="${HOST_TRAIN:-192.168.5.112}"
HOST_INFER="${HOST_INFER:-192.168.5.113}"
OUTDIR="${OUTDIR:-$EXP/results/run_$(date +%Y%m%d_%H%M%S)}"
ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-2}"
MODES="${MODES:-A,B,C,D}"
CONDA_LIB="$(dirname "$PY")/../lib"

mkdir -p "$OUTDIR"
# Sync code to infer node
tar cf - -C "$ROOT" experiments/reshard_real experiments/go_nogo/ib_counters.py | \
  ssh -o BatchMode=yes "$HOST_INFER" "mkdir -p '$ROOT' && tar xf - -C '$ROOT'"

# Verify matrix first (rank0 only check)
"$PY" "$EXP/gen_traffic_matrix.py" | tee "$OUTDIR/matrix_gen.log"
cp -f "$EXP/traffic_matrix.json" "$OUTDIR/" 2>/dev/null || true

cat > "$OUTDIR/hostfile" <<EOF
${HOST_TRAIN} slots=2
${HOST_INFER} slots=2
EOF

# NCCL knobs (INSTRUCTIONS §4). Do NOT use "=mlx5_1,=mlx5_3" — NCCL parses
# that as a single broken exact-match and silently drops mlx5_3.
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
export NCCL_NET_GDR_LEVEL=SYS
export GLOO_SOCKET_IFNAME=bond0
export OUTDIR

echo "OUTDIR=$OUTDIR modes=$MODES iters=$ITERS"
cat "$OUTDIR/hostfile"

set +e
"$MPIRUN" --allow-run-as-root --hostfile "$OUTDIR/hostfile" -np 4 --map-by slot \
  -x NCCL_PROTO -x NCCL_ALGO -x NCCL_IB_GID_INDEX \
  -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_NET_GDR_LEVEL \
  -x GLOO_SOCKET_IFNAME -x OUTDIR \
  bash -c '
    export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
    export RANK=${OMPI_COMM_WORLD_RANK:-0}
    export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
    export MASTER_ADDR='"${HOST_TRAIN}"'
    export MASTER_PORT='"${MASTER_PORT:-29561}"'
    export LD_LIBRARY_PATH="'"$CONDA_LIB"':${LD_LIBRARY_PATH:-}"
    # One NIC per local GPU (NUMA-local Active ports)
    if [[ "$LOCAL_RANK" == "0" ]]; then
      export NCCL_IB_HCA=mlx5_1
    else
      export NCCL_IB_HCA=mlx5_3
    fi
    cd "'"$EXP"'"
    export NCCL_DEBUG_FILE="'"$OUTDIR"'/nccl_rank${RANK}.log"
    exec "'"$PY"'" -u reshard_bench.py \
      --outdir "'"$OUTDIR"'" --iters '"$ITERS"' --warmup '"$WARMUP"' --modes '"$MODES"'
  ' 2>&1 | tee "$OUTDIR/run.log"
rc=${PIPESTATUS[0]}
set +u
set -e
echo "mpirun_rc=$rc" | tee -a "$OUTDIR/run.log"
exit "$rc"