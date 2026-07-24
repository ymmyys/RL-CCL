#!/usr/bin/env python3
"""Read InfiniBand / RoCE port counters (bytes are typically in 4-byte units)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PortCounters:
    hca: str
    port: int
    xmit_data: int
    rcv_data: int
    xmit_packets: int
    rcv_packets: int

    @property
    def xmit_bytes(self) -> int:
        # Linux IB sysfs counters are in 4-byte words
        return self.xmit_data * 4

    @property
    def rcv_bytes(self) -> int:
        return self.rcv_data * 4


def list_hcas() -> list[str]:
    base = Path("/sys/class/infiniband")
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def read_port(hca: str, port: int = 1) -> PortCounters | None:
    cdir = Path(f"/sys/class/infiniband/{hca}/ports/{port}/counters")
    if not cdir.is_dir():
        return None

    def _read(name: str) -> int:
        return int((cdir / name).read_text().strip())

    return PortCounters(
        hca=hca,
        port=port,
        xmit_data=_read("port_xmit_data"),
        rcv_data=_read("port_rcv_data"),
        xmit_packets=_read("port_xmit_packets"),
        rcv_packets=_read("port_rcv_packets"),
    )


def snapshot(hcas: list[str] | None = None, port: int = 1) -> dict[str, PortCounters]:
    if hcas is None:
        hcas = list_hcas()
    out: dict[str, PortCounters] = {}
    for hca in hcas:
        c = read_port(hca, port)
        if c is not None:
            out[hca] = c
    return out


def delta(before: dict[str, PortCounters], after: dict[str, PortCounters]) -> dict[str, dict]:
    keys = sorted(set(before) & set(after))
    result = {}
    for hca in keys:
        b, a = before[hca], after[hca]
        result[hca] = {
            "xmit_bytes": a.xmit_bytes - b.xmit_bytes,
            "rcv_bytes": a.rcv_bytes - b.rcv_bytes,
            "xmit_packets": a.xmit_packets - b.xmit_packets,
            "rcv_packets": a.rcv_packets - b.rcv_packets,
        }
    return result


def snapshot_json(hcas: list[str] | None = None) -> str:
    snap = snapshot(hcas)
    return json.dumps({k: asdict(v) for k, v in snap.items()}, indent=2)


if __name__ == "__main__":
    hcas = os.environ.get("IB_HCAS", "").split(",")
    hcas = [h for h in hcas if h] or None
    print(snapshot_json(hcas))
