# Planner vs baselines ¡ª fluid dataflow comparison

W = 14.19 GiB (Qwen2.5-7B BF16), NIC goodput 11.55 GB/s.
Times are fluid-model seconds (dataflow volume/balance only ¡ª NCCL
stack tax excluded; measured mode D carries ~2.5x constant on 2x2).

| Scenario | LB | ce-serial | p2p-direct | mode-d | planner | struct | vs LB |
|---|---:|---:|---:|---:|---:|---|---:|
| fanout N=2 rho=1.0 | 1.228 | 1.228 (1.0xW) | 1.228 (2.0xW) | 1.228 (1.0xW) | **1.228** (1.0xW) | star | 1.0 |
| fanout N=4 rho=1.0 | 1.228 | 1.228 (1.0xW) | 2.456 (4.0xW) | 1.228 (1.0xW) | **1.228** (1.0xW) | star | 1.0 |
| fanout N=8 rho=1.0 | 1.228 | 1.228 (1.0xW) | 4.913 (8.0xW) | 1.228 (1.0xW) | **1.228** (1.0xW) | star | 1.0 |
| fanout N=16 rho=1.0 | 1.228 | 1.228 (1.0xW) | 9.825 (16.0xW) | 1.228 (1.0xW) | **1.228** (1.0xW) | star | 1.0 |
| fanout N=8 rho=1.0 | 1.228 | 1.228 (1.0xW) | 4.913 (8.0xW) | 1.228 (1.0xW) | **1.228** (1.0xW) | star | 1.0 |
| fanout N=8 rho=0.5 | 1.638 | 2.456 (1.0xW) | 4.913 (8.0xW) | 2.456 (1.0xW) | **1.638** (2.67xW) | stripe | 1.0 |
| fanout N=8 rho=0.2 | 2.729 | 6.141 (1.0xW) | 4.913 (8.0xW) | 6.141 (1.0xW) | **2.729** (4.44xW) | stripe | 1.0 |
| fanout N=8 rho=0.0 | 4.913 | inf (1.0xW) | 4.913 (8.0xW) | inf (1.0xW) | **4.913** (8.0xW) | stripe | 1.0 |
| fanout N=4 fat-src(4NIC) | 1.228 | 1.228 (1.0xW) | 2.456 (4.0xW) | 1.228 (1.0xW) | **1.228** (1.0xW) | star | 1.0 |
| fanout N=2 2NIC-tgt | 0.614 | 1.228 (1.0xW) | 1.228 (2.0xW) | 0.614 (1.0xW) | **0.614** (1.0xW) | star | 1.0 |
| fanout N=8 2NIC-tgt | 0.614 | 1.228 (1.0xW) | 4.913 (8.0xW) | 0.614 (1.0xW) | **0.614** (1.0xW) | star | 1.0 |
| tp2->tp4 x2 inst | 0.614 | 1.228 (1.0xW) | 1.228 (2.0xW) | 0.614 (1.0xW) | **0.614** (1.0xW) | star | 1.0 |
| tp2->tp4 x4 inst | 0.614 | 1.228 (1.0xW) | 2.456 (4.0xW) | 0.614 (1.0xW) | **0.614** (1.0xW) | star | 1.0 |
