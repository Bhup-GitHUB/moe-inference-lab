package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"moe-inference-lab/internal/metrics"
	"moe-inference-lab/internal/pool"
)

func main() {
	mode := flag.String("mode", "baseline", "baseline, duplicate, split, or compare")
	experts := flag.Int("experts", 256, "number of experts")
	topk := flag.Int("topk", 8, "distinct experts per token")
	gpus := flag.Int("gpus", 32, "shared GPUs, or total GPUs for split defaults")
	tokens := flag.Int("tokens", 200000, "total prompt and output tokens")
	zipf := flag.Float64("zipf", 1.2, "Zipf exponent; 0 is uniform")
	duplicates := flag.Int("duplicate-top", 8, "number of hot experts to duplicate")
	seed := flag.Int64("seed", 1, "random seed")
	rate := flag.Float64("gpu-rate", 2000, "expert calls per simulated GPU per second")
	arrival := flag.Float64("arrival-rate", 8000, "offered tokens per simulated second")
	queue := flag.Int("queue", 16, "waiting jobs per GPU")
	interval := flag.Float64("balance-interval", 0.1, "balancing interval in simulated seconds")
	prefillGPUs := flag.Int("prefill-gpus", 0, "prefill GPU count (default: half of --gpus)")
	decodeGPUs := flag.Int("decode-gpus", 0, "decode GPU count (default: remaining --gpus)")
	prompt := flag.Int("prompt-tokens", 32, "prompt tokens per sequence")
	decode := flag.Int("decode-tokens", 8, "generated tokens per sequence")
	handoff := flag.Float64("handoff-ms", 1, "KV cache handoff delay in milliseconds")
	output := flag.String("csv", "results.csv", "CSV output path")
	flag.Parse()
	if flag.NArg() != 0 {
		fail("unexpected positional arguments")
	}
	if *mode != "baseline" && *mode != "duplicate" && *mode != "split" && *mode != "compare" {
		fail("unknown mode")
	}
	if *experts < 1 || *topk < 1 || *topk > *experts || *gpus < 1 || *gpus > *experts || *tokens < 1 || *queue < 1 || *duplicates < 0 || *prompt < 1 || *decode < 1 {
		fail("invalid counts: require 1 <= topk,gpus <= experts, positive tokens, queue and sequence lengths, and nonnegative duplicate-top")
	}
	for _, v := range []float64{*zipf, *rate, *arrival, *interval, *handoff} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			fail("numeric flags must be finite")
		}
	}
	if *zipf < 0 || *rate <= 0 || *arrival <= 0 || *interval <= 0 || *handoff < 0 {
		fail("zipf and handoff must be nonnegative; rates and interval must be positive")
	}
	if *prefillGPUs < 0 || *decodeGPUs < 0 {
		fail("split GPU counts cannot be negative")
	}
	if *prefillGPUs == 0 {
		*prefillGPUs = *gpus / 2
	}
	if *decodeGPUs == 0 {
		*decodeGPUs = *gpus - *prefillGPUs
	}
	if (*mode == "split" || *mode == "compare") && (*prefillGPUs < 1 || *decodeGPUs < 1 || *prefillGPUs > *experts || *decodeGPUs > *experts) {
		fail("split pool GPU counts must be between 1 and experts")
	}
	c := pool.Config{Experts: *experts, TopK: *topk, GPUs: *gpus, DuplicateTop: *duplicates, Queue: *queue, Skew: *zipf, Rate: *rate, ArrivalRate: *arrival, BalanceInterval: *interval, Seed: *seed}
	run := func(m string) metrics.Result {
		if m != "split" {
			return pool.Run(m, c, *tokens)
		}
		p, d := c, c
		p.GPUs, d.GPUs = *prefillGPUs, *decodeGPUs
		d.Seed++
		return pool.RunSplit(p, d, *tokens, *prompt, *decode, *handoff/1000)
	}
	if *mode == "compare" {
		fmt.Printf("%-11s %12s %11s %11s %10s\n", "MODE", "TOKENS/S", "P50 MS", "P99 MS", "MAX/AVG")
		for _, m := range []string{"baseline", "duplicate", "split"} {
			r := run(m)
			fmt.Printf("%-11s %12.1f %11.2f %11.2f %10.2f\n", r.Mode, r.Throughput, r.P50*1000, r.P99*1000, r.Imbalance)
			path := filepath.Join(filepath.Dir(*output), m+"-"+filepath.Base(*output))
			if err := metrics.WriteCSV(path, r); err != nil {
				fail(err.Error())
			}
		}
		return
	}
	r := run(*mode)
	metrics.Print(os.Stdout, r)
	if err := metrics.WriteCSV(*output, r); err != nil {
		fail(err.Error())
	}
}

func fail(message string) { fmt.Fprintln(os.Stderr, "error:", message); os.Exit(2) }
