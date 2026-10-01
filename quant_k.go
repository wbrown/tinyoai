package tinyoai

import "encoding/binary"

// kBlockBytes returns the encoded block size for a validated Q5_1, Q6_K, or
// Q5_K tensor. The default branch is Q5_K, not a general unknown-format
// fallback.
func kBlockBytes(dtype string) int {
	if dtype == "Q5_1" {
		return 24
	}
	if dtype == "Q6_K" {
		return 210
	}
	return 176 // Q5_K
}

// quantBlockWidth returns 32 weights for Q5_1 and 256 for a validated
// K-quantized format.
func quantBlockWidth(dtype string) int {
	if dtype == "Q5_1" {
		return 32
	}
	return 256
}

// decodeKBlock expands one validated GGML Q5_1, Q5_K, or Q6_K block into out.
// K blocks hold 256 weights; Q5_1 holds 32. Group scales and minima stay in
// their on-disk order, and separate float32 operations preserve reference
// rounding.
func decodeKBlock(out []float32, block []byte, dtype string) {
	// Clio's Q5_K_M file uses Q5_1 for some down projections, whose
	// 7552-element rows are not divisible by the 256-weight K block size.
	if dtype == "Q5_1" {
		// Bytes: d[2], minimum[2], high-bit plane[4], paired nibbles[16].
		// Each value is d*q + minimum, where q joins its low four bits to
		// one bit from the plane. The two nibbles encode positions 16 apart.
		d, minimum := halfFloat(binary.LittleEndian.Uint16(block)), halfFloat(binary.LittleEndian.Uint16(block[2:]))
		high := binary.LittleEndian.Uint32(block[4:])
		for i := 0; i < 32; i++ {
			low := (block[8+i%16] >> uint(i/16*4)) & 15
			q := low | byte((high>>uint(i))&1)<<4
			out[i] = float32(d*float32(q)) + minimum
		}
		return
	}
	if dtype == "Q6_K" {
		// Layout: low nibbles[128], high two-bit planes[64], signed group
		// scales[16], and a half-float block scale[2]. Each 16-value group
		// reconstructs d*scale*(q-32); the bias centers six unsigned bits.
		d := halfFloat(binary.LittleEndian.Uint16(block[208:]))
		for group := 0; group < 16; group++ {
			scale := d * float32(int8(block[192+group]))
			for j := 0; j < 16; j++ {
				i := group*16 + j
				low := (block[(i/128)*64+i%64] >> uint((i%128)/64*4)) & 15
				high := (block[128+(i/128)*32+i%32] >> uint((i%128)/32*2)) & 3
				out[i] = scale * float32(int(low|high<<4)-32)
			}
		}
		return
	}
	d := halfFloat(binary.LittleEndian.Uint16(block))
	dmin := halfFloat(binary.LittleEndian.Uint16(block[2:]))
	// Q5_K lays out d[2], dmin[2], packed scale/minimum pairs[12],
	// high-bit planes[32], then low nibbles[128]. Eight 32-value groups use
	// d*scale*q - dmin*minimum. Six-bit scale/minimum fields share the top
	// bits of the first eight metadata bytes with the final four groups.
	scales := block[4:16]
	for group := 0; group < 8; group++ {
		var scale, minimum byte
		if group < 4 {
			scale, minimum = scales[group]&63, scales[group+4]&63
		} else {
			scale = scales[group+4]&15 | (scales[group-4]>>6)<<4
			minimum = scales[group+4]>>4 | (scales[group]>>6)<<4
		}
		ds, dm := d*float32(scale), dmin*float32(minimum)
		for j := 0; j < 32; j++ {
			low := (block[48+(group/2)*32+j] >> uint(group%2*4)) & 15
			high := (block[16+j] >> uint(group)) & 1
			// Preserve the decoder's separate float32 multiply and subtract.
			out[group*32+j] = float32(ds*float32(low|high<<4)) - dm
		}
	}
}
