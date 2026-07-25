#!/usr/bin/env bash
# Day 1: ib_write_bw + nccl-tests broadcast busbw baselines (112↔113).
#
# Env (fixed for reproducibility — match experiments/reshard_real/run_reshard.sh):
#   NCCL: Simple+Ring, GID=3, bond0, per-GPU HCA pin (mlx5_1 / mlx5_3)
#   NCCL libs: Megatron_test conda (torch ships NCCL 2.28.9) OR nccl_inject build
#   ib_write_bw: /usr/bin/ib_write_bw, GID index 3, size 8MiB
#
# Usage: ./bench/nic_baseline.sh [OUTDIR]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${1:-$ROOT/bench/results/nic_baseline_$(date +%Y%m%d_%H%M%S)}"
HOST_TRAIN="${HOST_TRAIN:-192.168.5.112}"
HOST_INFER="${HOST_INFER:-192.168.5.113}"
MPIRUN="${MPIRUN:-/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpirun}"
NCCL_TESTS="${NCCL_TESTS:-/home/xiajinyi25/nccl_inject/nccl-tests/build}"
# nccl-tests linked against system CUDA 13 — do NOT prepend conda lib (broken libcudart.so.13)
CUDA_LIB="${CUDA_LIB:-/usr/local/cuda/lib64}"
NCCL_LIB="${NCCL_LIB:-/home/xiajinyi25/nccl_inject/nccl/build/lib}"
WORKER_LD="${CUDA_LIB}:${NCCL_LIB}"
MSG_BYTES="${MSG_BYTES:-15231233024}"   # W from Qwen2.5-7B BF16
IB_SIZE="${IB_SIZE:-8388608}"           # 8 MiB
GID="${GID:-3}"
ITERS="${ITERS:-20}"

mkdir -p "$OUTDIR"
exec > >(tee "$OUTDIR/run.log") 2>&1

echo "OUTDIR=$OUTDIR"
echo "date=$(date -Is)"
echo "host_train=$(hostname) ip=$HOST_TRAIN"
echo "WORKER_LD=$WORKER_LD MSG_BYTES=$MSG_BYTES"

# --- discover RoCE IPs (Active ports mlx5_1 / mlx5_3) ---
ip_of() {
  local host="$1" dev="$2"
  ssh -o BatchMode=yes "$host" "ip -o -4 addr show dev $dev 2>/dev/null | awk '{print \$4}' | cut -d/ -f1"
}
# Map HCA → netdev (from prior ENV: mlx5_1↔ens7f1np1 style; probe via ibdev2netdev)
mapfile -t TRAIN_MAP < <(ssh -o BatchMode=yes "$HOST_TRAIN" "ibdev2netdev")
mapfile -t INFER_MAP < <(ssh -o BatchMode=yes "$HOST_INFER" "ibdev2netdev")
echo "=== ibdev2netdev train ==="; printf '%s\n' "${TRAIN_MAP[@]}"
echo "=== ibdev2netdev infer ==="; printf '%s\n' "${INFER_MAP[@]}"

netdev_for() {
  local lines="$1" hca="$2"
  printf '%s\n' "$lines" | awk -v h="$hca" '$1==h {print $5; exit}'
}
TRAIN_MAP_S=$(printf '%s\n' "${TRAIN_MAP[@]}")
INFER_MAP_S=$(printf '%s\n' "${INFER_MAP[@]}")
DEV_T1=$(netdev_for "$TRAIN_MAP_S" mlx5_1)
DEV_T3=$(netdev_for "$TRAIN_MAP_S" mlx5_3)
DEV_I1=$(netdev_for "$INFER_MAP_S" mlx5_1)
DEV_I3=$(netdev_for "$INFER_MAP_S" mlx5_3)
IP_T1=$(ip_of "$HOST_TRAIN" "$DEV_T1")
IP_T3=$(ip_of "$HOST_TRAIN" "$DEV_T3")
IP_I1=$(ip_of "$HOST_INFER" "$DEV_I1")
IP_I3=$(ip_of "$HOST_INFER" "$DEV_I3")
echo "IPs: train mlx5_1=$IP_T1 ($DEV_T1) mlx5_3=$IP_T3 ($DEV_T3)"
echo "IPs: infer mlx5_1=$IP_I1 ($DEV_I1) mlx5_3=$IP_I3 ($DEV_I3)"

run_ib() {
  local label="$1" server_host="$2" server_dev="$3" client_host="$4" client_dev="$5" server_ip="$6"
  local slog="$OUTDIR/ib_${label}_server.log"
  local clog="$OUTDIR/ib_${label}_client.log"
  echo "=== ib_write_bw $label : $client_host/$client_dev → $server_host/$server_dev ($server_ip) ==="
  # Kill only the binary (NOT -f: that matches the ssh cmdline and suicides the session).
  ssh -o BatchMode=yes "$server_host" "killall -q ib_write_bw 2>/dev/null || true"
  sleep 0.5
  ssh -o BatchMode=yes "$server_host" \
    "nohup /usr/bin/ib_write_bw -d $server_dev -x $GID -s $IB_SIZE -n $ITERS --report_gbits \
       > /tmp/ib_srv_$label.log 2>&1 < /dev/null & sleep 1; pgrep -x ib_write_bw || (cat /tmp/ib_srv_$label.log; exit 1)"
  if [[ "$client_host" == "$HOST_TRAIN" || "$client_host" == "192.168.5.112" ]]; then
    /usr/bin/ib_write_bw -d "$client_dev" -x "$GID" -s "$IB_SIZE" -n "$ITERS" --report_gbits "$server_ip" | tee "$clog"
  else
    ssh -o BatchMode=yes "$client_host" \
      "/usr/bin/ib_write_bw -d $client_dev -x $GID -s $IB_SIZE -n $ITERS --report_gbits $server_ip" | tee "$clog"
  fi
  ssh -o BatchMode=yes "$server_host" "cat /tmp/ib_srv_$label.log; killall -q ib_write_bw 2>/dev/null || true" | tee "$slog"
  echo "LABEL=$label" >> "$OUTDIR/ib_summary.txt"
  # perftest table: last numeric column often avg BW (Gbps with --report_gbits)
  awk '/^[[:space:]]*[0-9]/ {line=$0} END{print line}' "$clog" | tee -a "$OUTDIR/ib_summary.txt"
}

: > "$OUTDIR/ib_summary.txt"
run_ib same_rail_1 "$HOST_INFER" mlx5_1 "$HOST_TRAIN" mlx5_1 "$IP_I1"
run_ib cross_rail_1to3 "$HOST_INFER" mlx5_3 "$HOST_TRAIN" mlx5_1 "$IP_I3"
run_ib same_rail_3 "$HOST_INFER" mlx5_3 "$HOST_TRAIN" mlx5_3 "$IP_I3"

# --- nccl-tests broadcast: 4 ranks (2+2), message ≈ W ---
cat > "$OUTDIR/hostfile" <<EOF
${HOST_TRAIN} slots=2
${HOST_INFER} slots=2
EOF

# Sync nothing needed — nccl-tests already on both nodes
# Do NOT export conda LD_LIBRARY_PATH into mpirun itself — orted breaks
# (OpenSSL version mismatch). Inject only inside the worker bash -c.
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_DEBUG=WARN

echo "=== nccl-tests broadcast_perf 4GPU (W-sized) ==="
set +e
"$MPIRUN" --allow-run-as-root --hostfile "$OUTDIR/hostfile" -np 4 --map-by slot \
  -x NCCL_PROTO -x NCCL_ALGO -x NCCL_IB_GID_INDEX \
  -x NCCL_SOCKET_IFNAME -x NCCL_NET_GDR_LEVEL -x NCCL_DEBUG \
  bash -c '
    export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
    export LD_LIBRARY_PATH="'"$WORKER_LD"':${LD_LIBRARY_PATH:-}"
    if [[ "$LOCAL_RANK" == "0" ]]; then export NCCL_IB_HCA=mlx5_1; else export NCCL_IB_HCA=mlx5_3; fi
    exec "'"$NCCL_TESTS"'/broadcast_perf" -b '"$MSG_BYTES"' -e '"$MSG_BYTES"' -f 2 -g 1 -c 0 -n 5 -w 2
  ' 2>&1 | tee "$OUTDIR/nccl_broadcast_W.log"
rc_w=$?
set -e

echo "=== nccl-tests broadcast_perf sweep 64MiB..4GiB ==="
set +e
"$MPIRUN" --allow-run-as-root --hostfile "$OUTDIR/hostfile" -np 4 --map-by slot \
  -x NCCL_PROTO -x NCCL_ALGO -x NCCL_IB_GID_INDEX \
  -x NCCL_SOCKET_IFNAME -x NCCL_NET_GDR_LEVEL -x NCCL_DEBUG \
  bash -c '
    export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
    export LD_LIBRARY_PATH="'"$WORKER_LD"':${LD_LIBRARY_PATH:-}"
    if [[ "$LOCAL_RANK" == "0" ]]; then export NCCL_IB_HCA=mlx5_1; else export NCCL_IB_HCA=mlx5_3; fi
    exec "'"$NCCL_TESTS"'/broadcast_perf" -b 64M -e 4G -f 2 -g 1 -c 0 -n 20 -w 5
  ' 2>&1 | tee "$OUTDIR/nccl_broadcast_sweep.log"
rc_s=$?
set -e

# Parse busbw (GB/s) from last data line of W-sized run
python3 - <<'PY' "$OUTDIR"
import re, sys, json
from pathlib import Path
outdir = Path(sys.argv[1])
def parse_nccl(path):
    text = path.read_text(errors="replace")
    # typical: size  count  type  redop  root  time  algbw  busbw  #wrong
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0].isdigit():
            try:
                rows.append({
                    "size": int(parts[0]),
                    "time_us": float(parts[5]),
                    "algbw_GBs": float(parts[6]),
                    "busbw_GBs": float(parts[7]),
                })
            except ValueError:
                pass
    return rows
w_rows = parse_nccl(outdir / "nccl_broadcast_W.log")
sw_rows = parse_nccl(outdir / "nccl_broadcast_sweep.log")
ib = {}
# Parse client logs: data row after header with BW average column
for p in sorted(outdir.glob("ib_*_client.log")):
    label = p.name[len("ib_"):-len("_client.log")]
    for line in p.read_text(errors="replace").splitlines():
        parts = line.split()
        # 8388608  20  92.52  92.52  0.001379
        if len(parts) >= 4 and parts[0].isdigit() and int(parts[0]) > 1000:
            try:
                ib[label] = float(parts[3])  # BW average[Gb/sec]
            except ValueError:
                pass

out = {
    "ib_write_bw_Gbps": ib,
    "nccl_broadcast_W": w_rows[-1] if w_rows else None,
    "nccl_broadcast_sweep": sw_rows,
}
(outdir / "baseline.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY

echo "DONE rc_w=$rc_w rc_s=$rc_s baseline=$OUTDIR/baseline.json"
exit 0
