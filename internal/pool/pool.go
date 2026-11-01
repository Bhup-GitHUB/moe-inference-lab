package pool

import (
	"container/heap"
	"moe-inference-lab/internal/balancer"
	"moe-inference-lab/internal/gpu"
	"moe-inference-lab/internal/metrics"
	"moe-inference-lab/internal/router"
)

type Config struct {
	Experts, TopK, GPUs, DuplicateTop, Queue int
	Skew, Rate, ArrivalRate, BalanceInterval float64
	Seed                                     int64
}

type Pool struct {
	name                       string
	config                     Config
	workers                    []*gpu.GPU
	router                     *router.Router
	balancer                   *balancer.Balancer
	experts                    []int
	dispatch, nextBalance, end float64
}

func New(name string, c Config) *Pool {
	p := &Pool{name: name, config: c, router: router.New(c.Experts, c.TopK, c.Skew, c.Seed), balancer: balancer.New(c.Experts, c.GPUs, c.DuplicateTop), nextBalance: c.BalanceInterval}
	for i := 0; i < c.GPUs; i++ {
		p.workers = append(p.workers, gpu.New(c.Queue, c.Rate))
	}
	return p
}

func (p *Pool) Token(arrival float64) (float64, float64) {
	start := max(arrival, p.dispatch)
	if p.config.DuplicateTop > 0 && start >= p.nextBalance {
		p.balancer.Rebalance()
		p.nextBalance = start + p.config.BalanceInterval
	}
	p.experts = p.router.Route(p.experts)
	dispatch, done := start, start
	for _, expert := range p.experts {
		p.balancer.Record(expert)
		worker := p.workers[p.balancer.Choose(expert)]
		accepted, finished := worker.Submit(dispatch)
		dispatch = accepted
		done = max(done, finished)
	}
	p.dispatch = dispatch
	p.end = max(p.end, done)
	return start, done
}

func (p *Pool) Stats(elapsed float64) []metrics.GPUStat {
	stats := make([]metrics.GPUStat, len(p.workers))
	for i, g := range p.workers {
		stats[i] = metrics.GPUStat{Pool: p.name, ID: i, Calls: g.Calls(), Util: float64(g.Calls()) / p.config.Rate / elapsed}
	}
	return stats
}
func (p *Pool) Close() {
	for _, g := range p.workers {
		g.Close()
	}
}

func Run(mode string, c Config, tokens int) metrics.Result {
	if mode == "baseline" {
		c.DuplicateTop = 0
	}
	p := New("shared", c)
	defer p.Close()
	latencies := make([]float64, 0, tokens)
	for i := 0; i < tokens; i++ {
		start, done := p.Token(float64(i) / c.ArrivalRate)
		latencies = append(latencies, done-start)
	}
	r := metrics.Result{Mode: mode, Tokens: tokens, Elapsed: p.end}
	r.P50, r.P99 = metrics.Percentiles(latencies)
	r.GPUs = p.Stats(r.Elapsed)
	r.Finish()
	return r
}

type sequence struct {
	id, remaining int
	ready, start  float64
	first         bool
}
type sequenceHeap []sequence

func (h sequenceHeap) Len() int { return len(h) }
func (h sequenceHeap) Less(i, j int) bool {
	if h[i].ready == h[j].ready {
		return h[i].id < h[j].id
	}
	return h[i].ready < h[j].ready
}
func (h sequenceHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *sequenceHeap) Push(v any)   { *h = append(*h, v.(sequence)) }
func (h *sequenceHeap) Pop() any     { old := *h; v := old[len(old)-1]; *h = old[:len(old)-1]; return v }

func RunSplit(prefill, decode Config, tokens, promptTokens, decodeTokens int, handoff float64) metrics.Result {
	p, d := New("prefill", prefill), New("decode", decode)
	defer p.Close()
	defer d.Close()
	latencies := make([]float64, 0, tokens)
	decodeLatencies := make([]float64, 0)
	ttfts := make([]float64, 0)
	ready := &sequenceHeap{}
	heap.Init(ready)
	remaining, id, offered := tokens, 0, 0
	for remaining > 0 {
		prompt := min(promptTokens, remaining)
		remaining -= prompt
		output := min(decodeTokens, remaining)
		remaining -= output
		arrival := float64(offered) / prefill.ArrivalRate
		offered += prompt
		seqStart := max(arrival, p.dispatch)
		last := seqStart
		for i := 0; i < prompt; i++ {
			start, done := p.Token(arrival)
			latencies = append(latencies, done-start)
			last = max(last, done)
		}
		if output > 0 {
			heap.Push(ready, sequence{id: id, remaining: output, ready: last + handoff, start: seqStart, first: true})
		}
		id++
	}
	for ready.Len() > 0 {
		s := heap.Pop(ready).(sequence)
		start, done := d.Token(s.ready)
		latency := done - s.ready
		latencies = append(latencies, latency)
		decodeLatencies = append(decodeLatencies, latency)
		if s.first {
			ttfts = append(ttfts, done-s.start)
			s.first = false
		}
		_ = start
		s.remaining--
		if s.remaining > 0 {
			s.ready = done
			heap.Push(ready, s)
		}
	}
	r := metrics.Result{Mode: "split", Tokens: tokens, Elapsed: max(p.end, d.end)}
	r.P50, r.P99 = metrics.Percentiles(latencies)
	r.TTFTP50, r.TTFTP99 = metrics.Percentiles(ttfts)
	r.DecodeP50, r.DecodeP99 = metrics.Percentiles(decodeLatencies)
	r.GPUs = append(p.Stats(r.Elapsed), d.Stats(r.Elapsed)...)
	r.Finish()
	return r
}
