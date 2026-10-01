//go:build goexperiment.simd && go1.27

package tinyoai

import (
	"encoding/binary"
	"fmt"
	"math"
	"simd"
)

// numericalBackend reports the actual vector width and reduction block,
// distinguishing software-emulated SIMD from hardware execution.
func numericalBackend() string {
	if simd.Emulated() {
		return "simd-emulated-float64-matvec"
	}
	return fmt.Sprintf("simd-%d-block%d", simd.VectorBitSize(), reductionBlock)
}

type matvecInput struct {
	values, even, odd []float32
	quad              [4][]float32
}

// prepareMatvec packs activation lanes once for the stored weight format, then
// shares them read-only across row workers. It borrows x and reuses scratch
// when large enough; the packed views remain valid until scratch is reused.
func prepareMatvec(x []float32, dtype string, scratch []float32) matvecInput {
	input := matvecInput{values: x}
	if dtype == "Q8_0" && !simd.Emulated() && simd.VectorBitSize() <= 256 {
		data := scratch
		if cap(data) < len(x) {
			data = make([]float32, len(x))
		} else {
			data = data[:len(x)]
		}
		for part := range input.quad {
			input.quad[part] = data[part*(len(x)/4) : (part+1)*(len(x)/4)]
			for j := range input.quad[part] {
				input.quad[part][j] = x[j*4+part]
			}
		}
	}
	if dtype == "BF16" && !simd.Emulated() {
		// Split once per matrix-vector call, shared read-only by row workers.
		// Each packed uint32 contains an even and an odd BF16 weight. This
		// layout avoids scalar weight conversion or architecture-specific
		// widening instructions inside the dot product.
		size := 2 * ((len(x) + 1) / 2)
		data := scratch
		if cap(data) < size {
			data = make([]float32, size)
		} else {
			data = data[:size]
		}
		input.even, input.odd = data[:len(data)/2], data[len(data)/2:]
		for i, v := range x {
			if i%2 == 0 {
				input.even[i/2] = v
			} else {
				input.odd[i/2] = v
			}
		}
		if len(x)%2 != 0 {
			input.odd[len(x)/2] = 0
		}
	}
	return input
}

// mulRows dispatches the assigned rows to the SIMD implementation, which
// retains a scalar fallback for unsupported formats and emulated targets.
func (w weightTensor) mulRows(out []float32, input matvecInput, start, end int) {
	mulRowsSIMD(w, out, input, start, end)
}

// mulRowsSIMD computes the assigned rows using bounded float32 vector
// reductions combined in float64. Packed BF16/F32 weights are decoded in
// registers; other quantized formats use small tiles. F16 and emulated SIMD
// use the scalar oracle.
func mulRowsSIMD(w weightTensor, out []float32, input matvecInput, start, end int) {
	if w.dtype == "Q8_0" && input.quad[0] != nil {
		mulQ8Rows(w, out, input, start, end)
		return
	}
	if (w.dtype == "Q8_0" || w.dtype == "Q5_K" || w.dtype == "Q6_K" || w.dtype == "Q5_1") && !simd.Emulated() {
		var unpacked [reductionBlock]float32
		cols := w.shape[1]
		for row := start; row < end; row++ {
			var total float64
			for j := 0; j < cols; j += reductionBlock {
				n := min(reductionBlock, cols-j)
				w.readValues(unpacked[:n], row*cols+j)
				total += float64(attentionDot(input.values[j:j+n], unpacked[:n]))
			}
			out[row] = float32(total)
		}
		return
	}
	// Hardware-backed Go SIMD targets are little-endian. On other targets
	// the scalar decoder preserves safetensors' little-endian interpretation
	// and avoids paying for software-emulated vector operations.
	if simd.Emulated() || (w.dtype != "BF16" && w.dtype != "F32") {
		w.mulRowsScalar(out, input.values, start, end)
		return
	}
	var zero simd.Float32s
	width := zero.Len()
	blockSize := max(reductionBlock, 2*width)
	lanes := make([]float32, width)
	mask := simd.BroadcastUint32s(0xffff0000)
	x, cols := input.values, w.shape[1]
	for rowIndex := start; rowIndex < end; rowIndex++ {
		var total float64
		j := 0
		if w.dtype == "BF16" {
			row := w.data[rowIndex*cols*2 : (rowIndex+1)*cols*2]
			for j+2*width <= cols {
				// Bound float32 reduction depth independently of row length;
				// combine the short vector reductions in float64.
				limit := min(j+blockSize, cols)
				var even, odd simd.Float32s
				for ; j+2*width <= limit; j += 2 * width {
					bits := simd.LoadUint8s(row[j*2:]).ReshapeToUint32s()
					e := bits.ShiftAllLeft(16).BitsToFloat32()
					o := bits.And(mask).BitsToFloat32()
					even = e.MulAdd(simd.LoadFloat32s(input.even[j/2:]), even)
					odd = o.MulAdd(simd.LoadFloat32s(input.odd[j/2:]), odd)
				}
				even.Store(lanes)
				for _, v := range lanes {
					total += float64(v)
				}
				odd.Store(lanes)
				for _, v := range lanes {
					total += float64(v)
				}
			}
			for ; j < cols; j++ {
				total += float64(x[j]) * float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(row[j*2:]))<<16))
			}
		} else {
			row := w.data[rowIndex*cols*4 : (rowIndex+1)*cols*4]
			for j+width <= cols {
				limit := min(j+blockSize, cols)
				var sum simd.Float32s
				for ; j+width <= limit; j += width {
					v := simd.LoadUint8s(row[j*4:]).ReshapeToUint32s().BitsToFloat32()
					sum = v.MulAdd(simd.LoadFloat32s(x[j:]), sum)
				}
				sum.Store(lanes)
				for _, v := range lanes {
					total += float64(v)
				}
			}
			for ; j < cols; j++ {
				total += float64(x[j]) * float64(math.Float32frombits(binary.LittleEndian.Uint32(row[j*4:])))
			}
		}
		out[rowIndex] = float32(total)
	}
}

// mulQ8Rows multiplies Q8_0 rows by activation lanes prepared in four groups.
// Each packed word yields four signed bytes in registers. The caller selects
// this path only for hardware vectors no wider than 256 bits, because a Q8_0
// block holds 32 weights.
func mulQ8Rows(w weightTensor, out []float32, input matvecInput, start, end int) {
	var zero simd.Float32s
	width, cols := zero.Len(), w.shape[1]
	lanes := make([]float32, width)
	for row := start; row < end; row++ {
		var total float64
		for begin := 0; begin < cols; begin += reductionBlock {
			var a, b, c, d simd.Float32s
			for j := begin; j < min(begin+reductionBlock, cols); j += 32 {
				block := w.data[(row*cols+j)/32*34:][:34]
				scale := simd.BroadcastFloat32s(halfFloat(binary.LittleEndian.Uint16(block)))
				for k := 0; k < 32; k += 4 * width {
					bits := simd.LoadUint8s(block[2+k:]).ReshapeToUint32s()
					v0 := bits.ShiftAllLeft(24).BitsToInt32().ShiftAllRight(24).ConvertToFloat32().Mul(scale)
					v1 := bits.ShiftAllLeft(16).BitsToInt32().ShiftAllRight(24).ConvertToFloat32().Mul(scale)
					v2 := bits.ShiftAllLeft(8).BitsToInt32().ShiftAllRight(24).ConvertToFloat32().Mul(scale)
					v3 := bits.BitsToInt32().ShiftAllRight(24).ConvertToFloat32().Mul(scale)
					at := (j + k) / 4
					a = v0.MulAdd(simd.LoadFloat32s(input.quad[0][at:]), a)
					b = v1.MulAdd(simd.LoadFloat32s(input.quad[1][at:]), b)
					c = v2.MulAdd(simd.LoadFloat32s(input.quad[2][at:]), c)
					d = v3.MulAdd(simd.LoadFloat32s(input.quad[3][at:]), d)
				}
			}
			a.Store(lanes)
			for _, v := range lanes {
				total += float64(v)
			}
			b.Store(lanes)
			for _, v := range lanes {
				total += float64(v)
			}
			c.Store(lanes)
			for _, v := range lanes {
				total += float64(v)
			}
			d.Store(lanes)
			for _, v := range lanes {
				total += float64(v)
			}
		}
		out[row] = float32(total)
	}
}

// attentionDot returns the query/key dot product using float32 vector
// accumulators and a float64 combination of lanes and the scalar tail. Its
// reduction order differs from the scalar implementation.
func attentionDot(q, k []float32) float32 {
	var sum simd.Float32s
	width := sum.Len()
	i := 0
	for ; i+width <= len(q); i += width {
		sum = simd.LoadFloat32s(q[i:]).MulAdd(simd.LoadFloat32s(k[i:]), sum)
	}
	lanes := make([]float32, width)
	sum.Store(lanes)
	var total float64
	for _, v := range lanes {
		total += float64(v)
	}
	for ; i < len(q); i++ {
		total += float64(q[i]) * float64(k[i])
	}
	return float32(total)
}

// attentionAdd accumulates score*values into out with vector multiply-adds and
// a scalar tail. Each output lane retains the caller's history order.
func attentionAdd(out, values []float32, score float32) {
	v := simd.BroadcastFloat32s(score)
	width := v.Len()
	i := 0
	for ; i+width <= len(out); i += width {
		v.MulAdd(simd.LoadFloat32s(values[i:]), simd.LoadFloat32s(out[i:])).Store(out[i:])
	}
	for ; i < len(out); i++ {
		out[i] += score * values[i]
	}
}
