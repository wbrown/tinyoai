package tinyoai

import (
	"encoding/binary"
	"math"
	"runtime"
	"sync"
)

// mulRowsScalar writes W*x for matrix rows in [start, end), decoding weights
// in their stored format. Float64 accumulation limits error from long
// sequential reductions without expanding the model. This path also serves as
// the numerical oracle for optional SIMD kernels.
func (w weightTensor) mulRowsScalar(out, x []float32, start, end int) {
	cols := w.shape[1]
	for i := start; i < end; i++ {
		// Accumulate in float64 to limit error from long, sequential dot
		// products. Weights stay compact and activations stay float32; only
		// the scalar reduction uses extra precision.
		var sum float64
		switch w.dtype {
		case "Q5_K", "Q6_K", "Q5_1":
			var decoded [256]float32
			for j := 0; j < cols; j += 256 {
				n := min(256, cols-j)
				w.readValues(decoded[:n], i*cols+j)
				for k, weight := range decoded[:n] {
					sum += float64(x[j+k]) * float64(weight)
				}
			}
		case "Q8_0":
			row := w.data[i*cols/32*34 : (i+1)*cols/32*34]
			for j := 0; j < cols; j += 32 {
				block := row[j/32*34:][:34]
				var dot float64
				for k, v := range x[j : j+32] {
					dot += float64(v) * float64(int8(block[2+k]))
				}
				sum += dot * float64(halfFloat(binary.LittleEndian.Uint16(block)))
			}
		case "BF16":
			row := w.data[i*cols*2 : (i+1)*cols*2]
			// Independent accumulators shorten the dependency chain without
			// changing weight/activation precision or parallel row ownership.
			var a, b, c, d float64
			j := 0
			for ; j+3 < len(x); j += 4 {
				block := row[j*2 : j*2+8]
				a += float64(x[j]) * float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(block))<<16))
				b += float64(x[j+1]) * float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(block[2:]))<<16))
				c += float64(x[j+2]) * float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(block[4:]))<<16))
				d += float64(x[j+3]) * float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(block[6:]))<<16))
			}
			sum = (a + b) + (c + d)
			for ; j < len(x); j++ {
				sum += float64(x[j]) * float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(row[j*2:]))<<16))
			}
		case "F32":
			row := w.data[i*cols*4 : (i+1)*cols*4]
			for j, v := range x {
				sum += float64(v) * float64(math.Float32frombits(binary.LittleEndian.Uint32(row[j*4:])))
			}
		default:
			for j, v := range x {
				sum += float64(v) * float64(w.at(i*cols+j))
			}
		}
		out[i] = float32(sum)
	}
}

// parallelRows partitions [0, rows) into disjoint ranges and waits for every
// callback. Work is an approximate operation count used to keep small jobs
// serial; worker count is bounded by GOMAXPROCS. Callbacks must restrict
// writes to their assigned outputs.
func parallelRows(rows, work int, calc func(start, end int)) {
	workers := min(runtime.GOMAXPROCS(0), rows)
	if work < 1<<20 || workers == 1 {
		calc(0, rows)
		return
	}
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		start, end := rows*worker/workers, rows*(worker+1)/workers
		wg.Add(1)
		go func() { defer wg.Done(); calc(start, end) }()
	}
	wg.Wait()
}
