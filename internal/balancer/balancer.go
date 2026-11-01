package balancer

import "sort"

type Balancer struct {
	experts, gpus, top int
	counts             []int64
	copies             [][]int
	next               []int
}

func New(experts, gpus, top int) *Balancer {
	b := &Balancer{experts: experts, gpus: gpus, top: top, counts: make([]int64, experts), copies: make([][]int, experts), next: make([]int, experts)}
	for e := range b.copies {
		b.copies[e] = []int{e * gpus / experts}
	}
	return b
}

func (b *Balancer) Record(expert int) { b.counts[expert]++ }

func (b *Balancer) Choose(expert int) int {
	copies := b.copies[expert]
	i := b.next[expert] % len(copies)
	b.next[expert]++
	return copies[i]
}

func (b *Balancer) Rebalance() {
	order := make([]int, b.experts)
	loads := make([]float64, b.gpus)
	for e := range order {
		order[e] = e
		b.copies[e] = b.copies[e][:1]
		loads[b.copies[e][0]] += float64(b.counts[e])
	}
	sort.Slice(order, func(i, j int) bool {
		if b.counts[order[i]] == b.counts[order[j]] {
			return order[i] < order[j]
		}
		return b.counts[order[i]] > b.counts[order[j]]
	})
	for _, e := range order[:min(b.top, b.experts)] {
		if b.counts[e] == 0 || b.gpus < 2 {
			break
		}
		base := b.copies[e][0]
		best := -1
		for g := range loads {
			if g != base && (best < 0 || loads[g] < loads[best]) {
				best = g
			}
		}
		half := float64(b.counts[e]) / 2
		loads[base] -= half
		loads[best] += half
		b.copies[e] = append(b.copies[e], best)
	}
	clear(b.counts)
}
