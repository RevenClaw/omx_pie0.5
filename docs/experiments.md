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
| 2026-10-29 | pi05_omx_lora_debug / dryrun / 199 (our checkpoint) | server, localhost:4000 | 148.0 ms | 149.1 ms | 149.3 ms | ✅ Output **(10, 5)**. Server total p95 147.7 ms, network ~1 ms. First call 20.6 s (JIT). Slower than DROID (124 ms); cause unverified (unmerged LoRA adapters suspected). Reported "model 31 ms" is dispatch time only: openpi's `policy_timing` stops before the GPU finishes (JAX async dispatch), so only totals are reliable. |
| 2026-10-27 | pi05_droid | server, localhost:4000 | 123.6 ms | 124.4 ms | 124.6 ms | Server total ~123 ms, network ~1 ms. (The logged "model 95 / overhead 27 ms" split is unreliable: openpi's `policy_timing` stops before the GPU finishes (JAX async dispatch), so only totals are reliable.) First call 18.2 s (JIT). Output (15, 8). `PREALLOCATE=false`. |

## Training runs

| Date | Config | Dataset (hash, #episodes) | Batch | Steps | Wall time | Peak GPU mem | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-29 | pi05_omx_lora_debug (exp `dryrun`) | local/omx_synthetic (20 synthetic episodes, ~4.6k frames) | 16 | 200 | 14 min 33 s (incl. ~1 min load + compile, 2 saves) | ~17.4 GiB (XLA remat log) | ✅ Phase 2 dry run. **~3.9 s/it** → 30k steps ≈ 32–33 h. Loss 0.096 → 0.043. Checkpoint ≈ 10.7 GiB (7.2 params + 3.5 train state), ~15 s blocking save. Debug config keeps only the latest checkpoint. |
| 2026-10-27 | memtest (pi05 LoRA: gemma_2b_lora + gemma_300m_lora, H=10, no EMA) | FakeDataset | 16 | 100 | 9.6 min incl. load + compile | **17.72 / 21.24 GiB (83%)** | ✅ Fits, MEM_FRACTION=0.9. s/it: see the 2026-10-29 run (~3.9 s/it). Estimated batch 32 ≈ 23.4 GiB → would not fit. |

## Robot evaluations

| Date | Checkpoint | Task | Trials | Success | Interventions | Notes |
|---|---|---|---|---|---|---|
