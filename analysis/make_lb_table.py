#!/usr/bin/env python3
"""Build analysis/lb_table.md from Day1 baseline.json + reshard_real summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

W_BYTES = 15_231_233_024
# Four-mode means from run_20260724_132308
DEFAULT_MODES = {
    "A": 2.3717789205722513,
    "B": 3.474230756610632,
    "C": 1.9369920210912823,
    "D": 1.676018943451345,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Day1 baseline.json")
    ap.add_argument("--torch-bcast", default="", help="torch_broadcast.json (preferred ceiling)")
    ap.add_argument("--summary", default="", help="optional reshard summary.json")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    base = json.loads(Path(args.baseline).read_text())
    modes = dict(DEFAULT_MODES)
    if args.summary:
        s = json.loads(Path(args.summary).read_text())
        for row in s.get("rows", []):
            modes[row["mode"]] = row["wall_mean_s"]

    ib = base.get("ib_write_bw_Gbps") or {}
    ib_vals = [v for v in ib.values() if isinstance(v, (int, float))]
    ib_gbps = min(ib_vals) if ib_vals else 92.56
    nic_GBs = ib_gbps / 8.0  # Gbps → GB/s
    dual_cut_GBs = 2 * nic_GBs
    t_cut_dual = W_BYTES / (dual_cut_GBs * 1e9)
    t_cut_single = W_BYTES / (nic_GBs * 1e9)

    nccl_w = base.get("nccl_broadcast_W") or {}
    busbw = nccl_w.get("busbw_GBs")
    algbw = nccl_w.get("algbw_GBs")
    ceiling_src = "nccl-tests"
    if args.torch_bcast and Path(args.torch_bcast).exists():
        tb = json.loads(Path(args.torch_bcast).read_text())
        busbw = tb.get("algbw_GBs") or tb.get("busbw_GBs")
        algbw = tb.get("algbw_GBs")
        ceiling_src = "torch+NCCL"
    t_nccl = None
    if algbw:
        t_nccl = W_BYTES / (algbw * 1e9)
    elif nccl_w.get("time_us"):
        t_nccl = nccl_w["time_us"] * 1e-6

    lines = []
    lines.append("# Lower-bound / efficiency table (Day 1)")
    lines.append("")
    lines.append(f"- W = {W_BYTES} B ({W_BYTES/2**30:.2f} GiB)")
    lines.append(f"- ib_write_bw (min observed) = **{ib_gbps:.2f} Gbps** → {nic_GBs:.2f} GB/s per NIC")
    lines.append(f"- Dual-NIC cut (fluid) T ≥ W/(2·nic) = **{t_cut_dual:.3f} s**")
    lines.append(f"- Single-NIC cut T ≥ W/nic = **{t_cut_single:.3f} s**")
    lines.append(f"- ib map: `{ib}`")
    if busbw is not None:
        extra = f", time≈{t_nccl:.3f}s" if t_nccl else ""
        lines.append(
            f"- NCCL broadcast ceiling ({ceiling_src}) @W: "
            f"algbw/busbw=**{busbw:.3f} GB/s**{extra}"
        )
    else:
        lines.append("- NCCL broadcast ceiling: **MISSING**")
    lines.append("")
    lines.append(
        "Efficiency uses payload goodput `W/wall` over the NCCL ceiling "
        f"({ceiling_src}; preferred = torch stack matching mode D)."
    )
    lines.append("")
    lines.append("| Mode | wall_s | goodput_GBs (W/t) | % of dual-NIC cut | % of NCCL busbw |")
    lines.append("|---|---:|---:|---:|---:|")
    for m, wall in modes.items():
        gp = (W_BYTES / wall) / 1e9
        pct_cut = 100.0 * gp / dual_cut_GBs
        pct_nccl = (100.0 * gp / busbw) if busbw else float("nan")
        nccl_s = f"{pct_nccl:.1f}%" if busbw else "n/a"
        lines.append(
            f"| {m} | {wall:.3f} | {gp:.2f} | {pct_cut:.1f}% | {nccl_s} |"
        )

    d_wall = modes["D"]
    d_gp = (W_BYTES / d_wall) / 1e9
    d_vs_nccl = (d_gp / busbw) if busbw else None
    lines.append("")
    lines.append("## Day1 go/no-go")
    if d_vs_nccl is None:
        lines.append("- **UNDECIDED**: no NCCL busbw baseline")
        verdict = "undecided"
    else:
        ok = d_vs_nccl >= 0.80
        lines.append(
            f"- Mode D goodput / NCCL busbw = **{d_vs_nccl*100:.1f}%** "
            f"(threshold 80%) → **{'PASS' if ok else 'FAIL'}**"
        )
        verdict = "pass" if ok else "fail"
    lines.append("")
    lines.append(
        "> Note: dual-NIC fluid cut is an upper bound on link capacity; NCCL "
        "busbw is the reachable collective ceiling on this stack. Do **not** "
        "claim D reaches LB_cut; report % of NCCL busbw / cut as above."
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    meta = {"verdict": verdict, "D_vs_nccl_busbw": d_vs_nccl, "ib_gbps": ib_gbps, "busbw": busbw}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
