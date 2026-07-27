"""Plan serialization — the wire format between the offline planner and the
on-cluster executor.

`rmcast_plan.plan()` runs on a laptop; its output Plan (plus the Topology and
the src/dst demand it was built for) is dumped to JSON here, shipped to the
cluster, and re-hydrated by `experiments/reshard_real/plan_executor.py`. Keeping
this out of `rmcast_plan.py` lets the planner stay pure-algorithm with no I/O.

The JSON is a faithful round-trip: `plan_from_dict(plan_to_dict(...))` yields an
equal Topology and Plan (verified in test_plan_io). Ops carry their dependency
edges (`after_send` / `after_bcast`) and the `fanout` discipline so the executor
never has to re-derive scheduling from the topology.

CLI:  python plan_io.py --demo 2x2 --structure auto --out plan_2x2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent))

from rmcast_plan import (  # noqa: E402
    BcastOp, CopyOp, DEMOS, Host, Plan, SendOp, Topology, build_demand_classes,
    plan,
)


# --------------------------------------------------------------- topology I/O

def topo_to_dict(topo: Topology) -> dict:
    return {
        "hosts": {
            h.name: {
                "gpus": list(h.gpus),
                "nics": dict(h.nics),
                "gpu_nic": {str(g): n for g, n in h.gpu_nic.items()},
                "intra_bw": h.intra_bw,
                "relay_frac": h.relay_frac,
            }
            for h in topo.hosts.values()
        }
    }


def topo_from_dict(d: dict) -> Topology:
    topo = Topology()
    for name, hd in d["hosts"].items():
        topo.add_host(Host(
            name=name,
            gpus=[int(g) for g in hd["gpus"]],
            nics={k: float(v) for k, v in hd["nics"].items()},
            gpu_nic={int(g): n for g, n in hd["gpu_nic"].items()},
            intra_bw=float(hd.get("intra_bw", 20.0)),
            relay_frac=float(hd.get("relay_frac", 1.0)),
        ))
    return topo


# ------------------------------------------------------------------- plan I/O

def _key(t: tuple[str, str]) -> str:
    return f"{t[0]}/{t[1]}"


def _unkey(s: str) -> tuple[str, str]:
    h, n = s.split("/", 1)
    return h, n


def plan_to_dict(topo: Topology, src: dict, dst: dict, p: Plan) -> dict:
    return {
        "format": "rmcast-plan/1",
        "topology": topo_to_dict(topo),
        "src": {str(g): [list(iv) for iv in ivs] for g, ivs in src.items()},
        "dst": {str(g): [list(iv) for iv in ivs] for g, ivs in dst.items()},
        "plan": {
            "structure": p.structure,
            "predicted_s": p.predicted_s,
            "classes": [
                {"intervals": [list(iv) for iv in c.intervals],
                 "nbytes": c.nbytes,
                 "holders": sorted(c.holders),
                 "dsts": sorted(c.dsts)}
                for c in p.classes
            ],
            "sends": [
                {"src": o.src, "dst": o.dst, "nbytes": o.nbytes, "nic": o.nic,
                 "classes": list(o.classes), "stripe": o.stripe}
                for o in p.sends
            ],
            "bcasts": [
                {"root": o.root, "members": list(o.members), "nbytes": o.nbytes,
                 "nic": o.nic, "classes": list(o.classes), "stripe": o.stripe,
                 "after_send": o.after_send, "fanout": o.fanout}
                for o in p.bcasts
            ],
            "copies": [
                {"src": o.src, "dsts": list(o.dsts), "nbytes": o.nbytes,
                 "after_send": o.after_send, "after_bcast": o.after_bcast}
                for o in p.copies
            ],
            "nic_egress": {_key(k): v for k, v in p.nic_egress.items()},
            "host_ingress": dict(p.host_ingress),
            "relay_egress": dict(p.relay_egress),
        },
    }


def plan_from_dict(d: dict) -> tuple[Topology, dict, dict, Plan]:
    if d.get("format") != "rmcast-plan/1":
        raise ValueError(f"unknown plan format {d.get('format')!r}")
    topo = topo_from_dict(d["topology"])
    src = {int(g): [tuple(iv) for iv in ivs] for g, ivs in d["src"].items()}
    dst = {int(g): [tuple(iv) for iv in ivs] for g, ivs in d["dst"].items()}
    pj = d["plan"]
    # classes are rebuilt from src/dst to guarantee they match the demand
    # (the serialized copy is a cross-check, not the source of truth)
    classes = build_demand_classes(src, dst)
    sends = [SendOp(o["src"], o["dst"], o["nbytes"], o["nic"],
                    tuple(o["classes"]), o["stripe"]) for o in pj["sends"]]
    bcasts = [BcastOp(o["root"], tuple(o["members"]), o["nbytes"], o["nic"],
                      tuple(o["classes"]), o["stripe"], o["after_send"],
                      o.get("fanout", "ring")) for o in pj["bcasts"]]
    copies = [CopyOp(o["src"], tuple(o["dsts"]), o["nbytes"],
                     o.get("after_send", -1), o.get("after_bcast", -1))
              for o in pj["copies"]]
    p = Plan(
        structure=pj["structure"],
        classes=classes,
        sends=sends,
        bcasts=bcasts,
        copies=copies,
        nic_egress={_unkey(k): v for k, v in pj["nic_egress"].items()},
        host_ingress=dict(pj["host_ingress"]),
        relay_egress=dict(pj["relay_egress"]),
        predicted_s=pj["predicted_s"],
    )
    return topo, src, dst, p


def write_plan(path: str, topo: Topology, src: dict, dst: dict, p: Plan) -> None:
    Path(path).write_text(json.dumps(plan_to_dict(topo, src, dst, p), indent=2))


def read_plan(path: str) -> tuple[Topology, dict, dict, Plan]:
    return plan_from_dict(json.loads(Path(path).read_text()))


# ------------------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", choices=sorted(DEMOS), default="2x2")
    ap.add_argument("--structure", choices=["auto", "star", "stripe"],
                    default="auto")
    ap.add_argument("--out", required=True, help="output plan JSON path")
    args = ap.parse_args()

    topo, src, dst = DEMOS[args.demo]()
    p = plan(topo, src, dst, structure=args.structure)
    write_plan(args.out, topo, src, dst, p)
    print(f"wrote {args.out}: structure={p.structure} "
          f"predicted={p.predicted_s:.3f}s "
          f"sends={len(p.sends)} bcasts={len(p.bcasts)} copies={len(p.copies)}")


if __name__ == "__main__":
    main()
