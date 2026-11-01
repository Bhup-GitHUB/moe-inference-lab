package metrics

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"text/tabwriter"
)

type GPUStat struct {
	Pool  string
	ID    int
	Calls int64
	Util  float64
}
type Result struct {
	Mode                                                       string
	Tokens                                                     int
	Elapsed, Throughput, P50, P99, MinUtil, MaxUtil, Imbalance float64
	TTFTP50, TTFTP99, DecodeP50, DecodeP99                     float64
	GPUs                                                       []GPUStat
}

func Percentiles(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}
	sort.Float64s(values)
	return values[(len(values)*50+99)/100-1], values[(len(values)*99+99)/100-1]
}

func (r *Result) Finish() {
	var sum float64
	if len(r.GPUs) > 0 {
		r.MinUtil = r.GPUs[0].Util
	}
	for _, g := range r.GPUs {
		r.MinUtil = min(r.MinUtil, g.Util)
		r.MaxUtil = max(r.MaxUtil, g.Util)
		sum += g.Util
	}
	if sum > 0 {
		r.Imbalance = r.MaxUtil / (sum / float64(len(r.GPUs)))
	}
	if r.Elapsed > 0 {
		r.Throughput = float64(r.Tokens) / r.Elapsed
	}
}

func Print(w io.Writer, r Result) {
	t := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(t, "MODE\tTOKENS\tTOKENS/S\tP50 MS\tP99 MS\tMIN UTIL\tMAX UTIL\tMAX/AVG")
	fmt.Fprintf(t, "%s\t%d\t%.1f\t%.2f\t%.2f\t%.1f%%\t%.1f%%\t%.2f\n", r.Mode, r.Tokens, r.Throughput, r.P50*1000, r.P99*1000, r.MinUtil*100, r.MaxUtil*100, r.Imbalance)
	if r.Mode == "split" {
		fmt.Fprintf(t, "TTFT p50/p99: %.2f / %.2f ms\tDecode p50/p99: %.2f / %.2f ms\n", r.TTFTP50*1000, r.TTFTP99*1000, r.DecodeP50*1000, r.DecodeP99*1000)
	}
	fmt.Fprintln(t, "POOL\tGPU\tEXPERT CALLS\tUTILIZATION")
	for _, g := range r.GPUs {
		fmt.Fprintf(t, "%s\t%d\t%d\t%.1f%%\n", g.Pool, g.ID, g.Calls, g.Util*100)
	}
	t.Flush()
}

func WriteCSV(path string, r Result) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	rows := [][]string{{"record", "mode", "pool", "gpu", "tokens", "elapsed_s", "tokens_per_s", "p50_ms", "p99_ms", "min_util", "max_util", "max_avg", "ttft_p50_ms", "ttft_p99_ms", "decode_p50_ms", "decode_p99_ms", "expert_calls", "gpu_util"}}
	flt := func(x float64) string { return strconv.FormatFloat(x, 'f', 6, 64) }
	rows = append(rows, []string{"summary", r.Mode, "", "", strconv.Itoa(r.Tokens), flt(r.Elapsed), flt(r.Throughput), flt(r.P50 * 1000), flt(r.P99 * 1000), flt(r.MinUtil), flt(r.MaxUtil), flt(r.Imbalance), flt(r.TTFTP50 * 1000), flt(r.TTFTP99 * 1000), flt(r.DecodeP50 * 1000), flt(r.DecodeP99 * 1000), "", ""})
	for _, g := range r.GPUs {
		rows = append(rows, []string{"gpu", r.Mode, g.Pool, strconv.Itoa(g.ID), "", "", "", "", "", "", "", "", "", "", "", "", strconv.FormatInt(g.Calls, 10), flt(g.Util)})
	}
	w.WriteAll(rows)
	err = w.Error()
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}
