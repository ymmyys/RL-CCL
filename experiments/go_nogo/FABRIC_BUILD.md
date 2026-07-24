# fabric-lib / fabric-debug — no-sudo build + 112↔113 run

## Built (userland only)
- Rust **1.91.0** (`rust-toolchain.toml`)
- GDRCopy lib → `~/.local` (no `gdrdrv` kmod; compile-time only)
- libfabric 1.22.0 → `~/.local` (`LIBFABRIC_HOME`)
- libclang from apt `.deb` extract → `~/.local/lib/libclang`
- CUDA 13.3 patch: `cumem.rs` uses `__bindgen_anon_1.id`
- Local fixes: skip DOWN verbs ports; default `FABRIC_GID_INDEX=3` (RoCEv2)

## Build
```bash
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export GDRAPI_HOME=$HOME/.local LIBFABRIC_HOME=$HOME/.local
export LIBCLANG_PATH=$HOME/.local/lib/libclang
export LD_LIBRARY_PATH=$HOME/.local/lib:$LIBCLANG_PATH
cd ~/ymy/RL-CCL/pplx-garden
cargo build --release -p fabric-lib -p fabric-debug
```

## Run (Active NICs: mlx5_1 / mlx5_3)
```bash
# 113 server
FABRIC_GID_INDEX=3 LD_LIBRARY_PATH=$HOME/.local/lib:/usr/local/cuda/targets/x86_64-linux/lib \
  ./target/release/fabric-debug 0,1 1
# 112 client — paste Main Address from server
FABRIC_GID_INDEX=3 LD_LIBRARY_PATH=... \
  ./target/release/fabric-debug 0,1 1 <server_addr_hex>
```

## Result (2026-07-24, `results/fabric_debug_20260724_082638/`)
| Mode | Peak | Notes |
|---|---:|---|
| Paged Write (2 NIC) | **185 Gbps** (92%) | 2×100G aggregate |
| Single Write | **92 Gbps** (92%) | one path |
| Imm Write | **5.6 µs** | verified |

Topology after Active filter: GPU0→mlx5_1, GPU1→mlx5_3, link_speed=200Gbps.
