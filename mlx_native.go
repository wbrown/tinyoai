//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"time"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

type mlxMatrix struct{ weight, scales, biases mx.Array }
type mlxLayer struct {
	norm, bias                 mx.Array
	q, k, v, o, gate, up, down mlxMatrix
	keys, values               mlxKV
	qkv, gateUp                mlxMatrix
}
type nativeMLX struct {
	ctx                 *mx.Context
	config              StableLMConfig
	weights             map[string]mx.Array
	lora                *mlxLoRA
	layers              []mlxLayer
	embed, head         mlxMatrix
	norm, bias          mx.Array
	offset, group, bits int
	kvBits              int
	probabilityBatch    int
	topKBlock           int
	reserveTokens       int
	fusedProjections    bool
	projectionsPacked   bool
	densePrefill        bool
	onPrefillChunk      func(MLXPrefillChunk)
	capturePath         string
	capturePrefix       int
	layerBatch          int
	onPrefillPhase      func(MLXPrefillPhase)
	outputRoot          bool
}

// LoadMLXNative loads a StableLM checkpoint stored as FP16 or affine 2-, 3-,
// 4-, 5-, 6-, or 8-bit group-64 MLX weights. It validates tensor names,
// shapes, and dtypes, evaluates weights, and starts with FP16 KV.
//
// Go owns the model and cache; the MLX C API evaluates tensors without Python.
// Call Close on the returned model to release native resources.
func LoadMLXNative(dir string) (*MLX, error) {
	m, err := newMLX(dir)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Quantization *struct {
			Group int    `json:"group_size"`
			Bits  int    `json:"bits"`
			Mode  string `json:"mode"`
		} `json:"quantization"`
	}
	if err = json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	q := cfg.Quantization
	group, bits := 0, 16
	if q != nil {
		if q.Group != 64 || !slices.Contains([]int{2, 3, 4, 5, 6, 8}, q.Bits) || q.Mode != "affine" {
			return nil, fmt.Errorf("native MLX requires FP16 or affine 2/3/4/5/6/8-bit group-64 MLX weights")
		}
		group, bits = q.Group, q.Bits
	}
	n := &nativeMLX{
		config: m.config, group: group, bits: bits, kvBits: 16,
		weights: make(map[string]mx.Array),
		// Exact hierarchical selection avoids sorting the full vocabulary.
		// Keep the original one-token sampling head; only saved prompt scores
		// use the larger batch. See the retained prefill experiment results.
		probabilityBatch: 64, topKBlock: 1024,
	}
	files, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, err
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no safetensors weights in %s", dir)
	}
	err = mx.Run(func() {
		n.ctx = mx.New()
		a := &mx.Arena{Context: n.ctx}
		defer a.Free()
		for _, path := range files {
			for name, tensor := range a.Load(path) {
				if _, exists := n.weights[name]; exists {
					mx.Fail("duplicate tensor %s", name)
				}
				n.weights[name] = tensor.Retain()
			}
		}
		c := n.config
		n.embed = n.matrix("model.embed_tokens", c.VocabSize, c.HiddenSize)
		n.head = n.matrix("lm_head", c.VocabSize, c.HiddenSize)
		n.norm = n.weight("model.norm.weight", mx.Float16, c.HiddenSize)
		n.bias = n.weight("model.norm.bias", mx.Float16, c.HiddenSize)
		head := c.HiddenSize / c.NumAttentionHeads
		for i := 0; i < c.NumHiddenLayers; i++ {
			p := fmt.Sprintf("model.layers.%d.", i)
			n.layers = append(n.layers, mlxLayer{
				norm: n.weight(p+"input_layernorm.weight", mx.Float16, c.HiddenSize),
				bias: n.weight(p+"input_layernorm.bias", mx.Float16, c.HiddenSize),
				q:    n.matrix(p+"self_attn.q_proj", c.HiddenSize, c.HiddenSize),
				k:    n.matrix(p+"self_attn.k_proj", c.NumKeyValueHeads*head, c.HiddenSize),
				v:    n.matrix(p+"self_attn.v_proj", c.NumKeyValueHeads*head, c.HiddenSize),
				o:    n.matrix(p+"self_attn.o_proj", c.HiddenSize, c.HiddenSize),
				gate: n.matrix(p+"mlp.gate_proj", c.IntermediateSize, c.HiddenSize),
				up:   n.matrix(p+"mlp.up_proj", c.IntermediateSize, c.HiddenSize),
				down: n.matrix(p+"mlp.down_proj", c.HiddenSize, c.IntermediateSize),
			})
		}
		want := 8 + (c.NumHiddenLayers * 23)
		if n.bits == 16 {
			want = 4 + c.NumHiddenLayers*9
		}
		if len(n.weights) != want {
			mx.Fail("got %d tensors, expected %d", len(n.weights), want)
		}
		all := make([]mx.Array, 0, len(n.weights))
		for _, w := range n.weights {
			all = append(all, w)
		}
		mx.Eval(all...)
	})
	if err != nil {
		_ = n.Close()
		return nil, err
	}
	m.native = n
	return m, nil
}

// weight borrows a named retained tensor after checking its scalar type and
// exact dimensions. Validation failures escape through the enclosing MLX Run
// call.
func (n *nativeMLX) weight(name string, dtype int, shape ...int) mx.Array {
	w, ok := n.weights[name]
	if !ok {
		mx.Fail("missing MLX tensor %s", name)
	}
	if !slices.Equal(w.Shape(), shape) || w.Dtype() != dtype {
		mx.Fail("invalid shape or dtype for %s: %v, dtype %d", name, w.Shape(), w.Dtype())
	}
	return w
}

// matrix validates and borrows a dense weight matrix or its packed affine
// data, scales, and biases.
func (n *nativeMLX) matrix(name string, rows, cols int) mlxMatrix {
	if n.bits == 16 {
		return mlxMatrix{weight: n.weight(name+".weight", mx.Float16, rows, cols)}
	}
	return mlxMatrix{n.weight(name+".weight", mx.Uint32, rows, cols*n.bits/32), n.weight(name+".scales", mx.Float16, rows, cols/n.group), n.weight(name+".biases", mx.Float16, rows, cols/n.group)}
}

// linear adds an optional low-rank correction to the original projection.
// Prefill, decoding, saved probabilities, and branches share this operation;
// the adapter never merges into or requantizes the base weight matrix.
func (n *nativeMLX) linear(a *mx.Arena, x mx.Array, w mlxMatrix) mx.Array {
	y := n.baseLinear(a, x, w)
	if n.activeLoRA() {
		if pair, ok := n.lora.pairs[w.weight]; ok {
			delta := a.Matmul(a.Matmul(x, a.Transpose(pair.a, 1, 0)), a.Transpose(pair.b, 1, 0))
			y = a.Add(y, a.Scale(delta, n.lora.factor))
		}
	}
	return y
}

// baseLinear preserves the original dense and quantized projection paths.
// A zero-strength adapter skips its graph entirely, retaining base numerics.
func (n *nativeMLX) baseLinear(a *mx.Arena, x mx.Array, w mlxMatrix) mx.Array {
	if n.bits == 16 {
		return a.Matmul(x, a.Transpose(w.weight, 1, 0))
	}
	if n.densePrefill && x.Shape()[1] >= 256 {
		// Diagnostic alternative for devices whose dense GEMM is faster than
		// fused Q6 GEMM. The dequantized matrix is a disposable graph temporary;
		// small vocabulary-head batches keep the quantized path.
		return a.Matmul(x, a.Transpose(a.Dequantize(w.weight, w.scales, w.biases, n.group, n.bits), 1, 0))
	}
	return a.Linear(x, w.weight, w.scales, w.biases, n.group, n.bits)
}

// reset releases all retained layer KV and sets the valid prefix length to
// zero. It leaves weights intact and requires the MLX lock.
func (n *nativeMLX) reset() {
	for i := range n.layers {
		l := &n.layers[i]
		l.keys.Free()
		l.values.Free()
		l.keys, l.values = mlxKV{}, mlxKV{}
	}
	n.offset = 0
}

// LimitCache discards KV if its retained capacity exceeds limit. Selecting a
// smaller context therefore releases the larger allocation even when the new
// prompt shares a prefix; cleared reports that invalidation.
func (n *nativeMLX) LimitCache(limit int) (cleared bool, err error) {
	err = mx.Run(func() {
		for _, layer := range n.layers {
			if layer.keys.data.Valid() && layer.keys.data.Shape()[2] > limit {
				n.reset()
				mx.ClearCache()
				cleared = true
				break
			}
		}
	})
	return
}

// Close releases KV, retained weight handles, and streams under the MLX lock.
func (n *nativeMLX) Close() error {
	return mx.Run(func() {
		n.reset()
		n.lora.free()
		n.lora = nil
		for _, w := range n.weights {
			w.Free()
		}
		n.weights = nil
		n.ctx.Close()
		n.ctx = nil
	})
}

// updateMLXCache writes x after prefix along the sequence axis of [batch,
// head, position, feature] storage. Growth keeps only the valid prefix and
// adds capacity in 256-position increments, enlarged by an optional
// reservation.
//
// The cache pointer receives an owned reference to the full capacity. The
// returned arena-owned view exposes only populated positions, keeping spare
// and abandoned suffix slots out of attention.
func updateMLXCache(a *mx.Arena, cache *mx.Array, x mx.Array, prefix int, reserve ...int) mx.Array {
	shape := x.Shape()
	length := shape[2]
	old := *cache
	current := old
	if !current.Valid() || prefix+length > current.Shape()[2] {
		shape[2] = ((length + 255) / 256) * 256
		if len(reserve) > 0 {
			shape[2] = max(shape[2], reserve[0]-prefix)
		}
		extra := a.Zeros(x.Dtype(), shape...)
		if current.Valid() {
			current = a.Concat(a.Slice(current, 2, 0, prefix), extra, 2)
		} else {
			current = extra
		}
	}
	current = a.Update(current, x, 2, prefix)
	*cache = current.Retain()
	old.Free()
	return a.Slice(current, 2, 0, prefix+length)
}

// Forward replaces a resident suffix with tokens and optionally returns the
// final full-vocabulary logit row. The caller holds the model gate.
func (n *nativeMLX) Forward(ctx context.Context, prefix int, tokens []int, wantLogits bool) (logits []float32, err error) {
	logits, _, err = n.forward(ctx, prefix, tokens, wantLogits, nil, 0)
	return
}

// ForwardWithLogprobs evaluates tokens and records each target's log
// probability plus count top alternatives, normalized over the full
// vocabulary. Targets align one-for-one with input positions; the final
// full-vocabulary logits are also returned.
func (n *nativeMLX) ForwardWithLogprobs(ctx context.Context, prefix int, tokens, targets []int, count int) ([]float32, []logprobRow, error) {
	if len(targets) != len(tokens) || count < 1 || count > 64 {
		return nil, nil, fmt.Errorf("invalid probability request")
	}
	for _, id := range targets {
		if id < 0 || id >= n.config.VocabSize {
			return nil, nil, fmt.Errorf("invalid probability target")
		}
	}
	return n.forward(ctx, prefix, tokens, true, targets, count)
}

// forward builds and evaluates one StableLM chunk with parallel attention and
// MLP residuals. KV uses [batch, head, position, feature] axes; rotary
// positions begin at prefix. The caller holds the model gate.
//
// The arena releases temporary handles before evaluation so MLX can reuse
// cache buffers. Optional probability scoring batches the vocabulary
// projection, while the sampling head keeps its one-token shape. Native
// failures reset KV. Cancellation arriving during evaluation is deferred until
// the caller records the completed chunk and its probability rows.
func (n *nativeMLX) forward(ctx context.Context, prefix int, tokens []int, wantLogits bool, targets []int, count int) (logits []float32, rows []logprobRow, err error) {
	if err = ctx.Err(); err != nil {
		return nil, nil, err
	}
	if prefix < 0 || prefix > n.offset || len(tokens) == 0 || prefix+len(tokens) > n.config.MaxPositionEmbeddings {
		return nil, nil, fmt.Errorf("invalid MLX cached prefix or context length")
	}
	for _, token := range tokens {
		if token < 0 || token >= n.config.VocabSize {
			return nil, nil, fmt.Errorf("invalid token %d", token)
		}
	}
	err = mx.Run(func() {
		started := time.Now()
		if n.capturePath != "" && prefix == n.capturePrefix && len(tokens) > 1 {
			mx.StartCapture(n.capturePath)
			defer mx.StopCapture()
		}
		a := &mx.Arena{Context: n.ctx}
		defer a.Free()
		c := n.config
		length := len(tokens)
		dim := c.HiddenSize / c.NumAttentionHeads
		ids := a.Tokens(tokens)
		e := n.embed
		x := a.Take(e.weight, ids)
		if n.bits != 16 {
			x = a.Dequantize(x, a.Take(e.scales, ids), a.Take(e.biases, ids), n.group, n.bits)
		}
		for i := range n.layers {
			l := &n.layers[i]
			if n.onPrefillPhase != nil && length > 1 {
				x = n.profileLayer(a, x, i, prefix, length)
				continue
			}
			h := a.Norm(x, l.norm, l.bias, c.LayerNormEps)
			var q, k, v mx.Array
			if n.fusedProjections && !n.activeLoRA() {
				qkv := n.linear(a, h, l.qkv)
				kv := c.NumKeyValueHeads * dim
				q = a.Slice(qkv, 2, 0, c.HiddenSize)
				k = a.Slice(qkv, 2, c.HiddenSize, c.HiddenSize+kv)
				v = a.Slice(qkv, 2, c.HiddenSize+kv, c.HiddenSize+2*kv)
			} else {
				q, k, v = n.linear(a, h, l.q), n.linear(a, h, l.k), n.linear(a, h, l.v)
			}
			q = a.Transpose(a.Reshape(q, 1, length, c.NumAttentionHeads, dim), 0, 2, 1, 3)
			k = a.Transpose(a.Reshape(k, 1, length, c.NumKeyValueHeads, dim), 0, 2, 1, 3)
			v = a.Transpose(a.Reshape(v, 1, length, c.NumKeyValueHeads, dim), 0, 2, 1, 3)
			q = a.Rope(q, int(float64(dim)*c.PartialRotaryFactor), prefix, c.RopeTheta)
			k = a.Rope(k, int(float64(dim)*c.PartialRotaryFactor), prefix, c.RopeTheta)
			cachedK := l.keys.update(a, k, prefix, n.kvBits, n.reserveTokens)
			cachedV := l.values.update(a, v, prefix, n.kvBits, n.reserveTokens)
			scale := math.Sqrt(1 / float64(dim))
			var attn mx.Array
			if n.kvBits == 8 && length == 1 && c.NumAttentionHeads == c.NumKeyValueHeads {
				attn = quantizedDecodeAttention(a, q, cachedK, cachedV, scale)
			} else {
				attn = a.Attention(q, cachedK.dense(a), cachedV.dense(a), scale, length > 1)
			}
			r := n.linear(a, a.Reshape(a.Transpose(attn, 0, 2, 1, 3), 1, length, c.HiddenSize), l.o)
			var gate, up mx.Array
			if n.fusedProjections && !n.activeLoRA() {
				both := n.linear(a, h, l.gateUp)
				gate, up = a.Slice(both, 2, 0, c.IntermediateSize), a.Slice(both, 2, c.IntermediateSize, 2*c.IntermediateSize)
			} else {
				gate, up = n.linear(a, h, l.gate), n.linear(a, h, l.up)
			}
			mlp := n.linear(a, a.SwiGLU(gate, up), l.down)
			x = a.Add(a.Add(x, r), mlp)
			if n.layerBatch > 0 && length > 1 && (i+1)%n.layerBatch == 0 && i+1 < len(n.layers) {
				x = a.Materialize(x)[0]
			}
		}
		var output, hidden mx.Array
		if count > 0 {
			hidden = a.Norm(x, n.norm, n.bias, c.LayerNormEps).Retain()
			defer hidden.Free()
		} else if wantLogits {
			x = a.Norm(x, n.norm, n.bias, c.LayerNormEps)
			output = a.Contiguous(a.Cast(n.linear(a, a.Slice(x, 1, length-1, length), n.head), mx.Float32)).Retain()
			defer output.Free()
		}
		all := make([]mx.Array, 0, 2*len(n.layers)+1)
		if output.Valid() {
			all = append(all, output)
		}
		if hidden.Valid() {
			all = append(all, hidden)
		}
		if !n.outputRoot || len(all) == 0 {
			for _, l := range n.layers {
				all = append(all, l.keys.arrays()...)
				all = append(all, l.values.arrays()...)
			}
		}
		// Release graph-building handles before evaluation so MLX can reuse
		// KV storage in place instead of copying the full cache each token.
		a.Free()
		mx.Eval(all...)
		transformDone := time.Now()
		if output.Valid() {
			logits = output.Floats()
		}
		if hidden.Valid() {
			// Bound vocabulary projection scratch; transfer only chosen/top scores.
			batch := n.probabilityBatch
			if batch == 0 {
				batch = 32
			}
			// Finish probability rows for this completed KV chunk even when
			// cancelled. Returning partial rows would force a full re-prefill.
			for start := 0; start < length; start += batch {
				end := min(start+batch, length)
				p := &mx.Arena{Context: n.ctx}
				func() {
					defer p.Free()
					z := p.Cast(n.linear(p, p.Slice(hidden, 1, start, end), n.head), mx.Float32)
					normalizer := p.LogSumExp(z)
					indices := p.BlockTopIndices(z, count, n.topKBlock)
					top := p.Contiguous(p.Subtract(p.TakeAlong(z, indices), normalizer))
					chosen := p.Contiguous(p.Subtract(p.TakeAlong(z, p.Reshape(p.Tokens(targets[start:end]), 1, end-start, 1)), normalizer))
					idArray := p.Contiguous(p.Cast(indices, mx.Float32))
					var last mx.Array
					if end == length {
						// Keep the sampling head at its original one-token shape.
						last = p.Contiguous(p.Cast(n.linear(p, p.Slice(hidden, 1, length-1, length), n.head), mx.Float32))
					}
					arrays := []mx.Array{top, chosen, idArray}
					if last.Valid() {
						arrays = append(arrays, last)
					}
					mx.Eval(arrays...)
					values, picks, ids := top.Floats(), chosen.Floats(), idArray.Floats()
					for i := 0; i < end-start; i++ {
						r := logprobRow{Chosen: float64(picks[i]), IDs: make([]int, count), Values: make([]float64, count)}
						for j := 0; j < count; j++ {
							r.IDs[j] = int(ids[i*count+j])
							r.Values[j] = float64(values[i*count+j])
						}
						rows = append(rows, r)
					}
					if last.Valid() {
						logits = last.Floats()
					}
				}()
			}
		}
		n.offset = prefix + length
		if n.onPrefillChunk != nil && length > 1 {
			n.onPrefillChunk(MLXPrefillChunk{Prefix: prefix, Tokens: length,
				TransformerSeconds: transformDone.Sub(started).Seconds(),
				ProbabilitySeconds: time.Since(transformDone).Seconds(),
				TotalSeconds:       time.Since(started).Seconds(), Logits: logits})
		}
	})
	if err != nil {
		_ = mx.Run(func() { n.reset() })
		return nil, nil, err
	}
	// Return a completed chunk even if cancellation arrived during the kernel.
	// The caller records its KV/probability rows before checking cancellation at
	// the next chunk boundary; otherwise a successful forward loses all reuse.
	return logits, rows, nil
}
