# moe-inference-lab

A small, deterministic Go simulation of mixture-of-experts inference across GPUs. It runs on a laptop without GPUs, model weights, or external dependencies. It is inspired by [DeepSeek's DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), published March 1, 2025. This is a simplified teaching model, not a reproduction or benchmark of DeepSeek's system.

## Run

Requires Go 1.22 or newer.

```sh
go run ./cmd/sim --mode baseline --experts 256 --topk 8 --gpus 32 \
  --tokens 200000 --zipf 1.2 --duplicate-top 8 --seed 1

go run ./cmd/sim --mode duplicate
go run ./cmd/sim --mode split
make compare
```

`make run-baseline`, `make run-duplicate`, and `make run-split` run the individual modes. Each writes `results.csv` and prints a summary followed by every GPU's call count and utilization. `make compare` prints a side-by-side table and writes `baseline-results.csv`, `duplicate-results.csv`, and `split-results.csv`. Use `--csv` to change the output filename; compare prefixes the filename with the mode.

Example from `make compare`, using the defaults and seed 1:

| Mode | Tokens/s | p50 latency | p99 latency | GPU max/avg utilization |
|---|---:|---:|---:|---:|
| baseline | 547.3 | 10.50 ms | 11.50 ms | 14.62 |
| duplicate | 1,062.4 | 9.50 ms | 11.00 ms | 7.53 |
| split | 861.6 | 9.50 ms | 11.50 ms | 9.28 |

These are simulated numbers, not laptop execution speed. In this workload, duplicating hot experts spreads calls across more GPUs, improving throughput and p99 while reducing imbalance. Split uses different resources per stage and is included to demonstrate the architecture; its numbers are not a direct measurement of the benefit of duplication.

## Model

The router samples `--topk` distinct experts without replacement from a Zipf-weighted distribution over expert IDs. `--zipf 0` is uniform; larger values concentrate traffic on low-numbered experts. Static placement assigns contiguous, approximately equal groups of experts to GPUs. Each GPU is a goroutine with a bounded channel and a simulated bounded FIFO of `--queue` waiting calls plus one in service. Each call takes `1 / --gpu-rate` simulated seconds. A full GPU queue blocks the dispatcher until space opens, applying backpressure to later tokens. Expert calls for a token can overlap across GPUs; its completion is the last call to finish.

In `duplicate` mode, each `--balance-interval` simulated seconds, the balancer ranks experts by calls observed in the preceding interval. It adds one copy of each of the `--duplicate-top` hottest experts to a different GPU, selecting destinations using estimated per-GPU load. Calls alternate between the original and the copy. Placement is recalculated each interval; existing queued calls still finish on their original GPU. There is no modeled copy, memory, network, or migration cost.

In `split` mode, separate prefill and decode pools each have their own GPUs, router and balancer. A sequence receives a burst of prompt tokens, followed by `--handoff-ms` of KV-cache handoff delay. Decode schedules one token at a time per sequence, with the next token ready only when the previous one completes. Sequences can overlap. The final sequence may be partial when `--tokens` is not a multiple of prompt plus decode length; if it contains only prompt tokens, it has no TTFT sample. If no sequence has a decode token, TTFT and decode metrics are zero.

A virtual clock drives all service, arrivals, balancing, and handoff timing. Worker goroutine scheduling and the actual wall-clock runtime do not affect the results. The seed controls the router, so the same flags and seed produce identical CSV output.

## Metrics

- Throughput is all completed prompt and decode tokens divided by simulated time from zero to the last completion.
- Token p50 and p99 are exact nearest-rank percentiles of simulated token latency. For the shared and prefill pools, latency starts when a token is admitted by the dispatcher and ends when all its expert calls finish; time waiting before dispatcher admission is excluded. For decode, latency starts when the sequence is ready for its next token and includes dispatcher waiting. Split's overall percentiles combine both populations.
- TTFT is measured from the first prompt token's admission until the first decode token completes, including prompt processing, handoff and decode waiting. Decode latency is reported separately for all generated tokens.
- GPU utilization is its expert-call service time divided by the full run's elapsed simulated time. Min, max and max/average cover every GPU; in split mode they cover both pools, including time a pool is idle. Counts include calls completed after placement changes.
- CSV contains one `summary` row and one `gpu` row per GPU. Utilizations are fractions from 0 to 1; latencies are in milliseconds.

## Options

Run `go run ./cmd/sim --help` for all flags. Useful settings include:

| Flag | Default | Meaning |
|---|---:|---|
| `--mode` | `baseline` | `baseline`, `duplicate`, `split`, or `compare` |
| `--experts`, `--topk` | 256, 8 | Expert count and distinct experts per token |
| `--gpus` | 32 | Shared pool size; split defaults to half for each pool |
| `--prefill-gpus`, `--decode-gpus` | 0, 0 | Override split pool sizes; 0 uses the split default |
| `--tokens`, `--seed` | 200000, 1 | Total work and random seed |
| `--zipf`, `--duplicate-top` | 1.2, 8 | Routing skew and number of expert copies |
| `--gpu-rate`, `--arrival-rate` | 2000, 8000 | Expert calls/GPU/s and offered tokens/s |
| `--queue`, `--balance-interval` | 16, 0.1 | Waiting calls per GPU and balancing interval in simulated seconds |
| `--prompt-tokens`, `--decode-tokens` | 32, 8 | Sequence sizes in split mode |
| `--handoff-ms` | 1 | KV-cache transfer delay |
