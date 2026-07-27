#!/usr/bin/env bash
# Run the rmcast planner->NCCL executor bridge on 112 (2 GPU) + 113 (2 GPU).
# Generates a plan JSON from a demo, ships code (incl. planner/) to the infer
# node, and runs plan_executor.py under the SAME NCCL env as run_reshard.sh so
# the measured wall/egress is directly comparable to the four-mode results.
#
#   DEMO=2x2 STRUCTURE=auto ./run_plan_exec.sh
#   DEMO=tp2_tp4 STRUCTURE=stripe ./run_plan_exec.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/home/xiajinyi25/Megatron_test/conda_env/bin/python}"
MPIRUN="${MPIRUN:-/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpirun}"
HOST_TRAIN="${HOST_TRAIN:-192.168.5.112}"
HOST_INFER="${HOST_INFER:-192.168.5.113}"
DEMO="${DEMO:-2x2}"
STRUCTURE="${STRUCTURE:-auto}"
OUTDIR="${OUTDIR:-$EXP/results/plan_${DEMO}_${STRUCTURE}_$(date +%Y%m%d_%H%M%S)}"
ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-2}"
CONDA_LIB="$(dirname "$PY")/../lib"

mkdir -p "$OUTDIR"
# 1) generate the plan on the head node (pure stdlib planner)
"$PY" "$ROOT/planner/plan_io.py" --demo "$DEMO" --structure "$STRUCTURE" \
  --out "$OUTDIR/plan.json" | tee "$OUTDIR/plan_gen.log"

# 2) sync code to infer node — planner/ is now a dependency of the executor
tar cf - -C "$ROOT" experiments/reshard_real experiments/go_nogo/ib_counters.py planner | \
  ssh -o BatchMode=yes "$HOST_INFER" "mkdir -p '$ROOT' && tar xf - -C '$ROOT'"
# ship the plan itself (generated only on the head node)
ssh -o BatchMode=yes "$HOST_INFER" "mkdir -p '$OUTDIR'"
scp -q "$OUTDIR/plan.json" "$HOST_INFER:$OUTDIR/plan.json"

cat > "$OUTDIR/hostfile" <<EOF
${HOST_TRAIN} slots=2
${HOST_INFER} slots=2
EOF

# NCCL knobs — identical to run_reshard.sh (INSTRUCTIONS §4)
export NCCL_PROTO=Simple NCCL_ALGO=Ring NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH
export NCCL_NET_GDR_LEVEL=SYS GLOO_SOCKET_IFNAME=bond0 OUTDIR

echo "OUTDIR=$OUTDIR demo=$DEMO structure=$STRUCTURE iters=$ITERS"
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
    export MASTER_PORT='"${MASTER_PORT:-29563}"'
    export LD_LIBRARY_PATH="'"$CONDA_LIB"':${LD_LIBRARY_PATH:-}"
    if [[ "$LOCAL_RANK" == "0" ]]; then export NCCL_IB_HCA=mlx5_1
    else export NCCL_IB_HCA=mlx5_3; fi
    cd "'"$EXP"'"
    export NCCL_DEBUG_FILE="'"$OUTDIR"'/nccl_rank${RANK}.log"
    exec "'"$PY"'" -u plan_executor.py \
      --plan "'"$OUTDIR"'/plan.json" --outdir "'"$OUTDIR"'" \
      --iters '"$ITERS"' --warmup '"$WARMUP"'
  ' 2>&1 | tee "$OUTDIR/run.log"
rc=${PIPESTATUS[0]}
set -e
echo "mpirun_rc=$rc" | tee -a "$OUTDIR/run.log"
exit "$rc"
