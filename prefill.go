package tinyoai

import (
	"context"
	"math"
)

type stableBatch struct {
	x, norm, attOut, proj, gate, up, q, k, v []float32
	packed, att                              []float32
}

// batchState lazily allocates token-major scratch for one prefill chunk and
// resizes its attention scratch when KV capacity changes. The returned buffers
// belong to s and are reused only while its generation owns the cache.
func (m *StableLM) batchState(s *stableState) *stableBatch {
	c := m.config
	d, h, kv := c.HiddenSize, c.IntermediateSize, c.HiddenSize/c.NumAttentionHeads*c.NumKeyValueHeads
	if s.batch == nil {
		alloc := func(width int) []float32 { return make([]float32, prefillBatch*width) }
		stride := (prefillBatch + batchWidth() - 1) / batchWidth() * batchWidth()
		s.batch = &stableBatch{x: alloc(d), norm: alloc(d), attOut: alloc(d), proj: alloc(d), gate: alloc(h), up: alloc(h), q: alloc(d), k: alloc(kv), v: alloc(kv), packed: make([]float32, stride*max(d, h))}
	}
	if len(s.batch.att) != prefillBatch*len(s.att) {
		s.batch.att = make([]float32, prefillBatch*len(s.att))
	}
	return s.batch
}

// prefill evaluates chunks layer by layer, reusing matrix tiles across tokens.
// forward remains the one-token decode path and numerical reference. Positions
// and attention bounds are absolute, including when extending a cached prefix.
func (m *StableLM) prefill(ctx context.Context, ids []int, start int, s *stableState) error {
	if len(ids) == 0 {
		return nil
	}
	c := m.config
	d, h, head := c.HiddenSize, c.IntermediateSize, c.HiddenSize/c.NumAttentionHeads
	kv := head * c.NumKeyValueHeads
	rotary := int(float64(head) * c.PartialRotaryFactor)
	capacity := len(s.att) / c.NumAttentionHeads
	for offset := 0; offset < len(ids); offset += prefillBatch {
		n := min(prefillBatch, len(ids)-offset)
		if err := ctx.Err(); err != nil {
			return err
		}
		if n == 1 {
			if err := m.forward(ctx, ids[offset], start+offset, s, offset+n == len(ids)); err != nil {
				return err
			}
			if s.prefillDone != nil {
				s.prefillDone(start + offset + n)
			}
			continue
		}
		b := m.batchState(s)
		phase := s.profile.start()
		for j, id := range ids[offset : offset+n] {
			m.embed.readValues(b.x[j*d:(j+1)*d], id*d)
		}
		phase = s.profile.mark("embedding", phase)
		for l, w := range m.layers {
			if err := ctx.Err(); err != nil {
				return err
			}
			for j := 0; j < n; j++ {
				layerNorm(b.norm[j*d:(j+1)*d], b.x[j*d:(j+1)*d], w.norm, w.bias, c.LayerNormEps)
			}
			phase = s.profile.mark("layer_norm", phase)
			w.q.mulBatch(b.q, b.norm, n, b.packed)
			w.k.mulBatch(b.k, b.norm, n, b.packed)
			w.v.mulBatch(b.v, b.norm, n, b.packed)
			phase = s.profile.mark("qkv_projections", phase)
			for j := 0; j < n; j++ {
				stableRotate(b.q[j*d:(j+1)*d], head, rotary, start+offset+j, c.RopeTheta)
				stableRotate(b.k[j*kv:(j+1)*kv], head, rotary, start+offset+j, c.RopeTheta)
			}
			phase = s.profile.mark("rotary", phase)
			for j := 0; j < n; j++ {
				for kh := 0; kh < c.NumKeyValueHeads; kh++ {
					at := (kh*capacity + start + offset + j) * head
					s.storeKV(l, at, b.k[j*kv+kh*head:j*kv+(kh+1)*head], b.v[j*kv+kh*head:j*kv+(kh+1)*head])
				}
			}
			phase = s.profile.mark("kv_write", phase)
			m.prepareKV(l, start+offset+n, s)
			phase = s.profile.mark("kv_decode", phase)
			m.attendBatch(l, start+offset, n, s, b)
			phase = s.profile.mark("attention", phase)
			w.o.mulBatch(b.proj, b.attOut, n, b.packed)
			phase = s.profile.mark("output_projection", phase)
			w.gate.mulBatch(b.gate, b.norm, n, b.packed)
			w.up.mulBatch(b.up, b.norm, n, b.packed)
			for i, v := range b.gate[:n*h] {
				b.gate[i] = (v / (1 + float32(math.Exp(-float64(v))))) * b.up[i]
			}
			w.down.mulBatch(b.attOut, b.gate, n, b.packed)
			for i := range b.x[:n*d] {
				b.x[i] = (b.x[i] + b.proj[i]) + b.attOut[i]
			}
			phase = s.profile.mark("mlp", phase)
		}
		if offset+n == len(ids) {
			copy(s.x, b.x[(n-1)*d:n*d])
			layerNorm(s.norm, s.x, m.norm, m.bias, c.LayerNormEps)
			phase = s.profile.mark("layer_norm", phase)
			m.head.mulScratch(s.logits, s.norm, s.matvecScratch)
			s.profile.mark("lm_head", phase)
		}
		if s.prefillDone != nil {
			s.prefillDone(start + offset + n)
		}
	}
	return ctx.Err()
}
