.PHONY: run-baseline run-duplicate run-split compare

run-baseline:
	go run ./cmd/sim --mode baseline

run-duplicate:
	go run ./cmd/sim --mode duplicate

run-split:
	go run ./cmd/sim --mode split

compare:
	go run ./cmd/sim --mode compare
