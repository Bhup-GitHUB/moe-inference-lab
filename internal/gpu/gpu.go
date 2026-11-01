package gpu

type request struct{ arrival float64 }
type response struct{ accepted, done float64 }

type GPU struct {
	requests  chan request
	responses chan response
	calls     int64
}

func New(queue int, rate float64) *GPU {
	g := &GPU{requests: make(chan request, queue), responses: make(chan response)}
	go g.run(queue, rate)
	return g
}

func (g *GPU) run(queue int, rate float64) {
	pending := make([]float64, queue+1)
	head, count := 0, 0
	available := 0.0
	for req := range g.requests {
		at := req.arrival
		for count > 0 && pending[head] <= at {
			head = (head + 1) % len(pending)
			count--
		}
		if count == len(pending) {
			at = pending[head]
			head = (head + 1) % len(pending)
			count--
		}
		start := max(at, available)
		available = start + 1/rate
		pending[(head+count)%len(pending)] = available
		count++
		g.responses <- response{accepted: at, done: available}
	}
}

func (g *GPU) Submit(at float64) (float64, float64) {
	g.requests <- request{arrival: at}
	r := <-g.responses
	g.calls++
	return r.accepted, r.done
}

func (g *GPU) Calls() int64 { return g.calls }
func (g *GPU) Close()       { close(g.requests) }
