package tinyoai

import (
	"encoding/binary"
	"math"
)

const prefillBatch = 128
const reductionBlock = 512

// readValues unpacks a small contiguous weight tile, never the whole model.
func (w weightTensor) readValues(out []float32, offset int) {
	switch w.dtype {
	case "BF16":
		data := w.data[offset*2 : (offset+len(out))*2]
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(data[i*2:])) << 16)
		}
	case "F32":
		data := w.data[offset*4 : (offset+len(out))*4]
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
	case "Q8_0":
		for i := 0; i < len(out); {
			index := offset + i
			block := w.data[index/32*34:][:34]
			scale := halfFloat(binary.LittleEndian.Uint16(block))
			n := min(32-index%32, len(out)-i)
			for j := 0; j < n; j++ {
				out[i+j] = scale * float32(int8(block[2+index%32+j]))
			}
			i += n
		}
	case "Q5_K", "Q6_K", "Q5_1":
		size, width := kBlockBytes(w.dtype), quantBlockWidth(w.dtype)
		var decoded [256]float32
		for i := 0; i < len(out); {
			index := offset + i
			decodeKBlock(decoded[:], w.data[index/width*size:][:size], w.dtype)
			n := min(width-index%width, len(out)-i)
			copy(out[i:i+n], decoded[index%width:index%width+n])
			i += n
		}
	default:
		for i := range out {
			out[i] = w.at(offset + i)
		}
	}
}

// mulBatch computes token-major out = x * W^T. The only packed activation
// buffer belongs to the request. Four rows share each activation load and a
// weight tile is reused across the prompt chunk. F64 combines short F32 sums.
func (w weightTensor) mulBatch(out, x []float32, n int, packed []float32) {
	if n == 1 {
		w.mulScratch(out, x, packed)
		return
	}
	rows, cols := w.shape[0], w.shape[1]
	stride := (n + batchWidth() - 1) / batchWidth() * batchWidth()
	packed = packed[:cols*stride]
	// Pack [token, feature] as [feature, padded token]. SIMD lanes now
	// represent separate tokens, so one decoded weight serves a whole lane
	// group. Padding is zero, and only the original n tokens are written back.
	clear(packed)
	for token := 0; token < n; token++ {
		for col, v := range x[token*cols : (token+1)*cols] {
			packed[col*stride+token] = v
		}
	}
	parallelRows((rows+3)/4, rows*cols*n, func(start, end int) {
		var tile [4 * reductionBlock]float32
		totals := make([]float64, 4*stride)
		for group := start; group < end; group++ {
			row, nr := group*4, min(4, rows-group*4)
			clear(totals)
			for col := 0; col < cols; col += reductionBlock {
				nk := min(reductionBlock, cols-col)
				if nr != 4 {
					clear(tile[:])
				}
				for r := 0; r < nr; r++ {
					w.readValues(tile[r*nk:(r+1)*nk], (row+r)*cols+col)
				}
				batchDot4(totals, packed[col*stride:], tile[:4*nk], stride, nk)
			}
			for r := 0; r < nr; r++ {
				for token := 0; token < n; token++ {
					out[token*rows+row+r] = float32(totals[r*stride+token])
				}
			}
		}
	})
}
