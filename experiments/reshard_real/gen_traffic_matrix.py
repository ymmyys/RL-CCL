#!/usr/bin/env python3
"""Qwen2.5-7B Megatron TP=2 traffic matrix (synthetic BF16 shapes, no HF load).

Verified architecture constants from INSTRUCTIONS.md — script asserts exact byte totals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


# --- Qwen2.5-7B (HF config, verified) ---
HIDDEN = 3584
N_LAYERS = 28
N_HEADS = 28
N_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE = 18944
VOCAB = 152064
TP = 2
DTYPE_NBYTES = 2  # BF16

# Expected totals (INSTRUCTIONS.md)
W_BYTES = 15_231_233_024
SHARD_BYTES = 7_615_820_800  # per rank including replicated norms


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    owner: int  # TP rank that holds this shard (0 or 1); norms owner=0 for send accounting
    replicated: bool = False

    @property
    def nelem(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.nelem * DTYPE_NBYTES


@dataclass
class Bucket:
    owner: int
    specs: list[TensorSpec]
    nbytes: int


def build_tensor_specs() -> list[TensorSpec]:
    specs: list[TensorSpec] = []
    # Per TP rank: 14 q + 2 k + 2 v heads → 2304 out features
    qkv_out = (N_HEADS // TP) * HEAD_DIM + 2 * (N_KV_HEADS // TP) * HEAD_DIM
    assert qkv_out == 2304
    mlp_h = INTERMEDIATE // TP  # 9472
    vocab_h = VOCAB // TP  # 76032

    for layer in range(N_LAYERS):
        for owner in (0, 1):
            specs.append(
                TensorSpec(f"layers.{layer}.qkv.weight", (qkv_out, HIDDEN), owner)
            )
            specs.append(TensorSpec(f"layers.{layer}.qkv.bias", (qkv_out,), owner))
            specs.append(
                TensorSpec(f"layers.{layer}.o_proj.weight", (HIDDEN, HIDDEN // TP), owner)
            )
            specs.append(
                TensorSpec(f"layers.{layer}.gate_proj.weight", (mlp_h, HIDDEN), owner)
            )
            specs.append(
                TensorSpec(f"layers.{layer}.up_proj.weight", (mlp_h, HIDDEN), owner)
            )
            specs.append(
                TensorSpec(f"layers.{layer}.down_proj.weight", (HIDDEN, mlp_h), owner)
            )
        # RMSNorms: replicated; send-accounted on rank0 only
        specs.append(
            TensorSpec(
                f"layers.{layer}.input_layernorm.weight",
                (HIDDEN,),
                owner=0,
                replicated=True,
            )
        )
        specs.append(
            TensorSpec(
                f"layers.{layer}.post_attention_layernorm.weight",
                (HIDDEN,),
                owner=0,
                replicated=True,
            )
        )

    for owner in (0, 1):
        specs.append(TensorSpec("embed_tokens.weight", (vocab_h, HIDDEN), owner))
        specs.append(TensorSpec("lm_head.weight", (vocab_h, HIDDEN), owner))

    specs.append(
        TensorSpec("model.norm.weight", (HIDDEN,), owner=0, replicated=True)
    )
    return specs


def verify_specs(specs: list[TensorSpec]) -> dict:
    by_owner = {0: 0, 1: 0}
    unique = 0
    for s in specs:
        by_owner[s.owner] += s.nbytes
        if s.replicated:
            # held on both ranks in memory accounting for SHARD_BYTES
            pass
        unique += s.nbytes

    # Each rank also holds a local copy of norms in memory
    norm_bytes = sum(s.nbytes for s in specs if s.replicated)
    shard0_held = by_owner[0]  # norms already in owner0
    shard1_held = by_owner[1] + norm_bytes  # rank1 also holds norms locally

    # Unique payload W = sum of all specs (norms once)
    w = unique
    assert w == W_BYTES, f"W mismatch: {w} != {W_BYTES}"
    assert shard0_held == SHARD_BYTES, f"shard0 {shard0_held} != {SHARD_BYTES}"
    assert shard1_held == SHARD_BYTES, f"shard1 {shard1_held} != {SHARD_BYTES}"
    assert shard0_held + shard1_held - norm_bytes == W_BYTES

    return {
        "W_bytes": w,
        "W_GiB": w / (1024**3),
        "shard_bytes": SHARD_BYTES,
        "norm_bytes": norm_bytes,
        "n_tensors": len(specs),
        "by_owner_send": by_owner,  # bytes each owner must inject (norms once on 0)
    }


def pack_buckets(specs: list[TensorSpec], bucket_bytes: int = 256 << 20) -> list[Bucket]:
    """Pack each owner's send tensors into ~bucket_bytes chunks (order preserved)."""
    buckets: list[Bucket] = []
    for owner in (0, 1):
        owned = [s for s in specs if s.owner == owner]
        cur: list[TensorSpec] = []
        cur_n = 0
        for s in owned:
            if cur and cur_n + s.nbytes > bucket_bytes:
                buckets.append(Bucket(owner=owner, specs=cur, nbytes=cur_n))
                cur, cur_n = [], 0
            cur.append(s)
            cur_n += s.nbytes
        if cur:
            buckets.append(Bucket(owner=owner, specs=cur, nbytes=cur_n))
    return buckets


def main():
    specs = build_tensor_specs()
    summary = verify_specs(specs)
    buckets = pack_buckets(specs)
    print("=== traffic matrix OK ===")
    print(f"W = {summary['W_bytes']} B ({summary['W_GiB']:.4f} GiB)")
    print(f"shard/rank held = {summary['shard_bytes']} B")
    print(f"norm (replicated) = {summary['norm_bytes']} B")
    print(f"tensors = {summary['n_tensors']}")
    print(f"send by owner0 = {summary['by_owner_send'][0]} B")
    print(f"send by owner1 = {summary['by_owner_send'][1]} B")
    print(f"buckets = {len(buckets)} (target ~256 MiB)")
    for i, b in enumerate(buckets):
        print(f"  bucket[{i}] owner={b.owner} nbytes={b.nbytes} n_tensors={len(b.specs)}")
    # machine-readable for bench
    import json
    from pathlib import Path

    out = {
        "summary": summary,
        "specs": [asdict(s) for s in specs],
        "buckets": [
            {
                "owner": b.owner,
                "nbytes": b.nbytes,
                "specs": [asdict(s) for s in b.specs],
            }
            for b in buckets
        ],
    }
    Path(__file__).resolve().parent.joinpath("traffic_matrix.json").write_text(
        json.dumps(out, indent=2)
    )
    print("wrote traffic_matrix.json")


if __name__ == "__main__":
    main()
