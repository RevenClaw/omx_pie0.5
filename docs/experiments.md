# Experiments log

One entry per environment change, training run or evaluation. Newest first.

## Environment

| Date | Machine | GPU / driver | OS | openpi commit | Notes |
|---|---|---|---|---|---|
| 2026-10-27 | iras-hub | A30 24 GiB | — | `215abfb` | `verify_env.py` all ✅ (1380 GB free, ffmpeg present). Policy port 4000 (8000 taken). |
| 2026-10-26 | iras-hub | A30 24 GiB, driver 595.91.07, CUDA 13.2 | _fill in_ | `215abfb` (2026-08-24) | Phase 0 setup. GPU was shared (janusvln process holding 21.6 GB). |

## Latency (Phase 1)

| Date | Checkpoint | Where measured | p50 | p95 | p99 | Notes |
|---|---|---|---|---|---|---|
| 2026-10-27 | pi05_droid | server, localhost:4000 | 123.6 ms | 124.4 ms | 124.6 ms | Model 95 ms, server overhead ~27 ms, network ~1 ms. First call 18.2 s (JIT). Output (15, 8). `PREALLOCATE=false`. |

## Training runs

| Date | Config | Dataset (hash, #episodes) | Batch | Steps | Wall time | Peak GPU mem | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-27 | memtest (pi05 LoRA: gemma_2b_lora + gemma_300m_lora, H=10, no EMA) | FakeDataset | 16 | 100 | 9.6 min incl. load + compile | **17.72 / 21.24 GiB (83%)** | ✅ Fits, MEM_FRACTION=0.9. s/it: _pending_. Estimated batch 32 ≈ 23.4 GiB → would not fit. |

## Robot evaluations

| Date | Checkpoint | Task | Trials | Success | Interventions | Notes |
|---|---|---|---|---|---|---|
