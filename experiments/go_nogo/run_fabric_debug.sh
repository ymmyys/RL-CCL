#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH="${HOME}/.local/lib:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"
BIN="${HOME}/ymy/RL-CCL/pplx-garden/target/release/fabric-debug"
exec "$BIN" "$@"
