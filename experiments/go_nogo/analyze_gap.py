#!/usr/bin/env python3
"""Analyze go/no-go raw_results.json → gap table + verdict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(raw: dict) -> dict:
    W = raw["payload_bytes"]
    iters = raw["iters"]
    n_infer = raw["n_infer"]
    lb_relay = raw["lower_bound"]["with_relay_bytes_per_iter"]
    lb_p2p = raw["lower_bound"]["p2p_naive_bytes_per_iter"]

    # Identify train rank (rank 0) and sum infer-side if needed
    by_mode: dict[str, list] = {}
    for rank_results in raw["ranks"]:
        for r in rank_results:
            by_mode.setdefault(r["mode"], []).append(r)

    rows = []
    for mode, entries in by_mode.items():
        train = next(e for e in entries if e["rank"] == 0)
        xmit_total = train["xmit_bytes"]
        xmit_per_iter = xmit_total / max(iters, 1)
        # Primary lower bound for structural claim:
        #   broadcast should approach lb_relay (W)
        #   p2p should approach lb_p2p (N*W)
        lb = lb_relay if mode == "broadcast" else lb_p2p
        # Also compare both modes against the *optimal* (relay) lower bound
        gap_vs_own = xmit_per_iter / lb - 1.0 if lb else float("nan")
        gap_vs_opt = xmit_per_iter / lb_relay - 1.0 if lb_relay else float("nan")
        rows.append(
            {
                "mode": mode,
                "train_host": train["host"],
                "train_xmit_GB_total": xmit_total / 1e9,
                "train_xmit_GB_per_iter": xmit_per_iter / 1e9,
                "W_GB": W / 1e9,
                "N": n_infer,
                "lb_own_GB": lb / 1e9,
                "lb_opt_GB": lb_relay / 1e9,
                "gap_vs_own": gap_vs_own,
                "gap_vs_opt": gap_vs_opt,
                "wall_s_per_iter": train["wall_s_per_iter"],
                "infer_rcv": [
                    {
                        "rank": e["rank"],
                        "host": e["host"],
                        "rcv_GB_per_iter": e["rcv_bytes"] / max(iters, 1) / 1e9,
                        "xmit_GB_per_iter": e["xmit_bytes"] / max(iters, 1) / 1e9,
                    }
                    for e in entries
                    if e["rank"] != 0
                ],
            }
        )

    # Verdict on structural inefficiency of P2P vs optimal relay bound
    p2p = next((r for r in rows if r["mode"] == "p2p"), None)
    bcast = next((r for r in rows if r["mode"] == "broadcast"), None)
    verdict = "INCONCLUSIVE"
    reason = ""
    if p2p is not None:
        gap = p2p["gap_vs_opt"]
        if gap < 0.20:
            verdict = "NO-GO"
            reason = f"P2P train egress only {gap*100:.1f}% above relay lower bound (<20%)"
        elif gap > 0.50:
            verdict = "GO"
            reason = f"P2P train egress {gap*100:.1f}% above relay lower bound (>50%)"
        else:
            verdict = "BORDERLINE"
            reason = f"P2P gap={gap*100:.1f}% in [20%,50%]"

    # Structural claim: broadcast near W, p2p near N*W
    structural = {}
    if bcast and p2p:
        structural = {
            "bcast_xmit_over_W": bcast["train_xmit_GB_per_iter"] / (W / 1e9),
            "p2p_xmit_over_W": p2p["train_xmit_GB_per_iter"] / (W / 1e9),
            "p2p_over_bcast": p2p["train_xmit_GB_per_iter"] / max(bcast["train_xmit_GB_per_iter"], 1e-12),
            "expected_N": n_infer,
        }

    return {
        "W_bytes": W,
        "N_infer": n_infer,
        "iters": iters,
        "rows": rows,
        "structural": structural,
        "verdict": verdict,
        "reason": reason,
        "decision_rule": "gap_vs_opt = train_xmit/W - 1; <20% NO-GO, >50% GO",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_json", type=str)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    raw = json.loads(Path(args.raw_json).read_text())
    summary = summarize(raw)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)

    print("\n======== GO/NO-GO SUMMARY ========", flush=True)
    print(f"W={summary['W_bytes']/1e9:.3f} GB  N_infer={summary['N_infer']}  iters={summary['iters']}")
    for r in summary["rows"]:
        print(
            f"  {r['mode']:10s}  train_xmit/iter={r['train_xmit_GB_per_iter']:.3f} GB  "
            f"gap_vs_opt={r['gap_vs_opt']*100:.1f}%  wall={r['wall_s_per_iter']*1e3:.1f} ms"
        )
    if summary["structural"]:
        s = summary["structural"]
        print(
            f"  structural: bcast/W={s['bcast_xmit_over_W']:.2f}x  "
            f"p2p/W={s['p2p_xmit_over_W']:.2f}x  p2p/bcast={s['p2p_over_bcast']:.2f}x  "
            f"(expect ~{s['expected_N']}x)"
        )
    print(f"VERDICT: {summary['verdict']} — {summary['reason']}")


if __name__ == "__main__":
    main()
