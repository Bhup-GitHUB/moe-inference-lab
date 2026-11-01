package router

import (
	"math"
	"math/rand"
	"sort"
)

type Router struct {
	rng        *rand.Rand
	cdf        []float64
	topK       int
	marks      []uint64
	generation uint64
	skew       float64
}

func New(experts, topK int, skew float64, seed int64) *Router {
	r := &Router{rng: rand.New(rand.NewSource(seed)), cdf: make([]float64, experts), topK: topK, marks: make([]uint64, experts), skew: skew}
	for i := range r.cdf {
		r.cdf[i] = math.Pow(float64(i+1), -skew)
		if i > 0 {
			r.cdf[i] += r.cdf[i-1]
		}
	}
	return r
}

func (r *Router) Route(dst []int) []int {
	dst = dst[:0]
	r.generation++
	attempts := 0
	for len(dst) < r.topK {
		attempts++
		if attempts > 64*r.topK {
			i := r.remaining()
			r.marks[i] = r.generation
			dst = append(dst, i)
			continue
		}
		x := r.rng.Float64() * r.cdf[len(r.cdf)-1]
		i := sort.Search(len(r.cdf), func(i int) bool { return r.cdf[i] > x })
		if i == len(r.cdf) || r.marks[i] == r.generation {
			continue
		}
		r.marks[i] = r.generation
		dst = append(dst, i)
	}
	return dst
}

func (r *Router) remaining() int {
	first := 0
	for r.marks[first] == r.generation {
		first++
	}
	total := 0.0
	for i := first; i < len(r.cdf); i++ {
		if r.marks[i] != r.generation {
			total += math.Pow(float64(first+1)/float64(i+1), r.skew)
		}
	}
	x := r.rng.Float64() * total
	for i := first; i < len(r.cdf); i++ {
		if r.marks[i] != r.generation {
			x -= math.Pow(float64(first+1)/float64(i+1), r.skew)
			if x <= 0 {
				return i
			}
		}
	}
	return first
}
