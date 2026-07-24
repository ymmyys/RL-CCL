#!/usr/bin/env bash
# Go/No-Go launcher: 1 train rank on 112 + 2 infer ranks on 113.
# Usage:
#   bash experiments/go_nogo/run_go_nogo.sh
#   SIZE_GB=4 ITERS=5 bash experiments/go_nogo/run_go_nogo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/home/xiajinyi25/Megatron_test/conda_env/bin/python}"
MPIRUN="${MPIRUN:-/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpirun}"

HOST_TRAIN="${HOST_TRAIN:-192.168.5.112}"
HOST_INFER="${HOST_INFER:-192.168.5.113}"
IB_HCA="${NCCL_IB_HCA:-mlx5_1}"
SIZE_GB="${SIZE_GB:-2}"
ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-2}"
OUTDIR="${OUTDIR:-$ROOT/experiments/go_nogo/results/run_$(date +%Y%m%d_%H%M%S)}"

# Sync experiment code to infer node (homes are not shared)
echo "[sync] -> $HOST_INFER"
tar cf - -C "$ROOT" experiments/go_nogo | ssh -o BatchMode=yes "$HOST_INFER" \
  "mkdir -p '$ROOT' && tar xf - -C '$ROOT'"

mkdir -p "$OUTDIR"
HOSTFILE="$OUTDIR/hostfile"
# 1 slot on train, 2 slots on infer → N_infer=2
cat > "$HOSTFILE" <<EOF
${HOST_TRAIN} slots=1
${HOST_INFER} slots=2
EOF

echo "OUTDIR=$OUTDIR"
echo "topology: train=$HOST_TRAIN(1gpu) infer=$HOST_INFER(2gpu) HCA=$IB_HCA SIZE=${SIZE_GB}GiB"
echo "hostfile:"
cat "$HOSTFILE"

# NCCL/RoCE knobs (match Megatron_test/scripts). Do NOT export conda into
# LD_LIBRARY_PATH at mpirun level — it breaks OpenMPI/SSH (OpenSSL mismatch).
export NCCL_IB_HCA="$IB_HCA"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-SYS}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export OUTDIR
CONDA_LIB="$(dirname "$PY")/../lib"

# OpenMPI sets OMPI_COMM_WORLD_LOCAL_RANK; our script also reads LOCAL_RANK
set +e
"$MPIRUN" --allow-run-as-root --hostfile "$HOSTFILE" -np 3 --map-by slot \
  -x NCCL_IB_HCA -x NCCL_SOCKET_IFNAME -x NCCL_IB_GID_INDEX \
  -x NCCL_NET_GDR_LEVEL -x NCCL_DEBUG -x GLOO_SOCKET_IFNAME -x OUTDIR \
  bash -c '
    export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
    export RANK=${OMPI_COMM_WORLD_RANK:-0}
    export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
    export MASTER_ADDR='"${HOST_TRAIN}"'
    export MASTER_PORT='"${MASTER_PORT:-29551}"'
    export LD_LIBRARY_PATH="'"$CONDA_LIB"':${LD_LIBRARY_PATH:-}"
    cd "'"$EXP"'"
    exec "'"$PY"'" -u weight_xfer_bench.py \
      --mode both --size-gb '"$SIZE_GB"' --iters '"$ITERS"' --warmup '"$WARMUP"' \
      --hca "'"$IB_HCA"'" --outdir "'"$OUTDIR"'"
  ' 2>&1 | tee "$OUTDIR/run.log"
rc=${PIPESTATUS[0]}
set -e

if [[ $rc -ne 0 ]]; then
  echo "ERROR: mpirun failed rc=$rc" >&2
  tail -40 "$OUTDIR/run.log" >&2 || true
  exit $rc
fi

"$PY" "$EXP/analyze_gap.py" "$OUTDIR/raw_results.json" --out "$OUTDIR/summary.json" \
  | tee "$OUTDIR/summary.txt"

echo "Done. Artifacts in $OUTDIR"
