package tinyoai

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// StableLMConfig describes the Clio-compatible StableLM architecture. This
// backend supports parallel residuals, shared LayerNorm, SiLU, unscaled partial
// RoPE, and untied embeddings. Other StableLM variants are rejected on load.
type StableLMConfig struct {
	// ModelType must be "stablelm".
	ModelType string `json:"model_type"`
	// VocabSize is the number of embedding rows and output logits, including
	// unused padding IDs.
	VocabSize int `json:"vocab_size"`
	// HiddenSize is the residual-stream width.
	HiddenSize int `json:"hidden_size"`
	// IntermediateSize is the gated MLP width.
	IntermediateSize int `json:"intermediate_size"`
	// NumHiddenLayers is the number of transformer blocks.
	NumHiddenLayers int `json:"num_hidden_layers"`
	// NumAttentionHeads is the number of query heads; it must divide HiddenSize.
	NumAttentionHeads int `json:"num_attention_heads"`
	// NumKeyValueHeads is the number of KV heads; it must divide
	// NumAttentionHeads.
	NumKeyValueHeads int `json:"num_key_value_heads"`
	// HiddenAct must be "silu" for the implemented gated MLP.
	HiddenAct string `json:"hidden_act"`
	// MaxPositionEmbeddings is the total supported prompt-plus-forward position
	// limit.
	MaxPositionEmbeddings int `json:"max_position_embeddings"`
	// LayerNormEps is the positive stabilizer added to variance before inverse
	// square root.
	LayerNormEps float64 `json:"layer_norm_eps"`
	// RopeTheta is the positive frequency base for unscaled rotary embeddings.
	RopeTheta float64 `json:"rope_theta"`
	// PartialRotaryFactor selects the rotated fraction of each head; it must
	// produce a positive even width.
	PartialRotaryFactor float64 `json:"partial_rotary_factor"`
	// UseParallelResidual must be true: attention and MLP consume the same
	// normalized input.
	UseParallelResidual bool `json:"use_parallel_residual"`
	// UseQKVBias must be false because biased Q/K/V projections are not
	// implemented.
	UseQKVBias bool `json:"use_qkv_bias"`
	// QKLayerNorm must be false because per-head Q/K normalization is not
	// implemented.
	QKLayerNorm bool `json:"qk_layernorm"`
	// TieWordEmbeddings must be false because this backend expects a separate
	// output head.
	TieWordEmbeddings bool `json:"tie_word_embeddings"`
	// BOSTokenID is prepended by the tokenizer and included in StableLM prompt
	// usage.
	BOSTokenID int `json:"bos_token_id"`
	// EOSTokenID stops generation without emitting text, but still counts as a
	// sampled token.
	EOSTokenID int `json:"eos_token_id"`
	// RopeParameters accepts the newer nested representation; validation copies
	// it into the flat rotary fields.
	RopeParameters *struct {
		// Type must be "default"; scaled rotary variants are unsupported.
		Type string `json:"rope_type"`
		// Theta supplies the nested frequency base.
		Theta float64 `json:"rope_theta"`
		// Partial supplies the nested rotary fraction of each head.
		Partial float64 `json:"partial_rotary_factor"`
	} `json:"rope_parameters,omitempty"`
	// RopeScaling must be absent or JSON null; context-extension scaling is not
	// implemented.
	RopeScaling json.RawMessage `json:"rope_scaling,omitempty"`
}

// validate checks the supported StableLM architecture and normalizes nested
// RoPE parameters into the flat fields used by inference. Dimensions,
// special-token IDs, and rotary settings must be valid before allocating
// weights or KV.
func (c *StableLMConfig) validate() error {
	if c.ModelType != "stablelm" || !c.UseParallelResidual || c.UseQKVBias || c.QKLayerNorm || c.TieWordEmbeddings || c.HiddenAct != "silu" {
		return fmt.Errorf("unsupported StableLM variant: require parallel residuals, no QKV bias/QK norm, untied embeddings, and silu")
	}
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.NumKeyValueHeads <= 0 || c.VocabSize <= 0 || c.MaxPositionEmbeddings <= 0 ||
		c.HiddenSize%c.NumAttentionHeads != 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return fmt.Errorf("invalid StableLM dimensions")
	}
	if c.RopeParameters != nil {
		if c.RopeParameters.Type != "default" {
			return fmt.Errorf("unsupported RoPE type %q", c.RopeParameters.Type)
		}
		c.RopeTheta = c.RopeParameters.Theta
		c.PartialRotaryFactor = c.RopeParameters.Partial
	}
	if len(c.RopeScaling) > 0 && string(c.RopeScaling) != "null" {
		return fmt.Errorf("RoPE scaling is not supported")
	}
	rotary := float64(c.HiddenSize/c.NumAttentionHeads) * c.PartialRotaryFactor
	if !(c.LayerNormEps > 0) || math.IsInf(c.LayerNormEps, 0) || !(c.RopeTheta > 0) || math.IsInf(c.RopeTheta, 0) || !(c.PartialRotaryFactor > 0) || c.PartialRotaryFactor > 1 || rotary != math.Trunc(rotary) || int(rotary)%2 != 0 || rotary < 2 {
		return fmt.Errorf("invalid LayerNorm or partial RoPE settings")
	}
	if c.BOSTokenID < 0 || c.BOSTokenID >= c.VocabSize || c.EOSTokenID < 0 || c.EOSTokenID >= c.VocabSize {
		return fmt.Errorf("invalid BOS/EOS token IDs")
	}
	return nil
}

type stableLayer struct {
	norm, bias     weightTensor
	q, k, v, o     weightTensor
	gate, up, down weightTensor
}

// StableLM is a pure-Go CPU backend for NovelAI/clio-v1-legacy. Weights stay in
// their on-disk precision. Generate calls share one retained KV cache
// and take turns using it; concurrent callers may cancel while waiting.
type StableLM struct {
	config                  StableLMConfig
	tokenizer               *nerdstashTokenizer
	embed, head, norm, bias weightTensor
	layers                  []stableLayer
	cache                   chan *stableContext
	halfKV                  bool
	cacheIdentity           [32]byte
}

// LoadStableLM loads config.json, tokenizer.json, and single-file or sharded
// safetensors from a local Hugging Face model directory, or a GGUF file with
// tokenizer.json alongside it. It never downloads files or executes model code.
// BF16, F16, F32, and Clio's published GGUF Q8/Q6/Q5 variants are supported.
func LoadStableLM(dir string) (*StableLM, error) {
	if strings.EqualFold(filepath.Ext(dir), ".gguf") {
		return loadStableGGUF(dir)
	}
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	c := StableLMConfig{LayerNormEps: 1e-5, RopeTheta: 10000, PartialRotaryFactor: 0.25, HiddenAct: "silu"}
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("config: %w", err)
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	tok, err := loadNerdstash(filepath.Join(dir, "tokenizer.json"), c.VocabSize, c.BOSTokenID)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	store, err := openTensorStore(dir)
	if err != nil {
		return nil, err
	}
	defer store.close()
	return loadStableWeights(c, tok, store.load)
}

// loadStableWeights assembles a StableLM from a validated configuration and a
// shape-checking tensor reader. Both safetensors and GGUF feed this path; the
// GGUF adapter translates checkpoint names at its boundary. The cache channel
// starts with one ownership token and no allocated state.
func loadStableWeights(c StableLMConfig, tok *nerdstashTokenizer, read func(string, ...int) (weightTensor, error)) (*StableLM, error) {
	var err error
	m := &StableLM{config: c, tokenizer: tok, layers: make([]stableLayer, c.NumHiddenLayers)}
	load := func(name string, shape ...int) weightTensor {
		if err != nil {
			return weightTensor{}
		}
		var w weightTensor
		w, err = read(name, shape...)
		return w
	}
	d, h := c.HiddenSize, c.IntermediateSize
	kv := d / c.NumAttentionHeads * c.NumKeyValueHeads
	m.embed = load("model.embed_tokens.weight", c.VocabSize, d)
	m.head = load("lm_head.weight", c.VocabSize, d)
	m.norm = load("model.norm.weight", d)
	m.bias = load("model.norm.bias", d)
	for i := range m.layers {
		p := fmt.Sprintf("model.layers.%d.", i)
		m.layers[i] = stableLayer{
			norm: load(p+"input_layernorm.weight", d), bias: load(p+"input_layernorm.bias", d),
			q: load(p+"self_attn.q_proj.weight", d, d), k: load(p+"self_attn.k_proj.weight", kv, d),
			v: load(p+"self_attn.v_proj.weight", kv, d), o: load(p+"self_attn.o_proj.weight", d, d),
			gate: load(p+"mlp.gate_proj.weight", h, d), up: load(p+"mlp.up_proj.weight", h, d), down: load(p+"mlp.down_proj.weight", d, h),
		}
	}
	if err != nil {
		return nil, err
	}
	m.cache = make(chan *stableContext, 1)
	m.cache <- nil
	return m, nil
}

// Config returns an independent copy of the loaded architecture configuration,
// including nested rotary parameters and scaling metadata.
func (m *StableLM) Config() StableLMConfig {
	c := m.config
	if c.RopeParameters != nil {
		r := *c.RopeParameters
		c.RopeParameters = &r
	}
	c.RopeScaling = append(json.RawMessage(nil), c.RopeScaling...)
	return c
}

// Encode returns Clio token IDs, including the tokenizer's leading BOS token.
func (m *StableLM) Encode(prompt string) []int { return m.tokenizer.encode(prompt) }

type stableState struct {
	x, norm, attOut, proj, gate, up, q, k, v, att, logits []float32
	keys, values                                          [][]float32
	keys16, values16                                      [][]uint16
	keyWorkspace, valueWorkspace                          []float32
	matvecScratch                                         []float32
	profile                                               *stableProfile
	batch                                                 *stableBatch
	prefillDone                                           func(int)
}

// Profiling is opt-in for checkpoint comparisons. A normal generation never
// reads the clock. All marks run after the corresponding workers have joined.
type stableProfile map[string]time.Duration

// start records a profiling boundary, or returns the zero time when profiling
// is disabled. The ordinary inference path therefore performs no clock reads.
func (p *stableProfile) start() time.Time {
	if p == nil {
		return time.Time{}
	}
	return time.Now()
}

// mark adds elapsed time to stage and returns the next profiling boundary. A
// nil profile is a no-op, so the same forward pass can serve normal requests
// and diagnostic measurements.
func (p *stableProfile) mark(stage string, start time.Time) time.Time {
	if p == nil {
		return time.Time{}
	}
	now := time.Now()
	(*p)[stage] += now.Sub(start)
	return now
}

// newState allocates activation scratch and layer KV for capacity positions.
// KV is head-major, with capacity determining the stride between heads.
// Float16 storage also reserves one reusable float32 layer workspace for
// attention.
func (m *StableLM) newState(capacity int) *stableState {
	c := m.config
	d := c.HiddenSize
	kv := d / c.NumAttentionHeads * c.NumKeyValueHeads
	s := &stableState{
		x: make([]float32, d), norm: make([]float32, d), attOut: make([]float32, d), proj: make([]float32, d),
		gate: make([]float32, c.IntermediateSize), up: make([]float32, c.IntermediateSize),
		q: make([]float32, d), k: make([]float32, kv), v: make([]float32, kv),
		att: make([]float32, capacity*c.NumAttentionHeads), logits: make([]float32, c.VocabSize),
		keys: make([][]float32, c.NumHiddenLayers), values: make([][]float32, c.NumHiddenLayers),
		matvecScratch: make([]float32, (max(d, c.IntermediateSize)+1)&^1),
	}
	if m.halfKV {
		s.keys, s.values = nil, nil
		s.keys16, s.values16 = make([][]uint16, c.NumHiddenLayers), make([][]uint16, c.NumHiddenLayers)
		s.keyWorkspace, s.valueWorkspace = make([]float32, capacity*kv), make([]float32, capacity*kv)
		for i := range s.keys16 {
			s.keys16[i], s.values16[i] = make([]uint16, capacity*kv), make([]uint16, capacity*kv)
		}
	} else {
		for i := range s.keys {
			s.keys[i] = make([]float32, capacity*kv)
			s.values[i] = make([]float32, capacity*kv)
		}
	}
	return s
}

// layerNorm writes affine mean/variance normalization of x to out. Mean and
// variance use float64 reductions to limit cancellation, while the affine
// output remains float32. Weight and bias must match the nonempty activation
// width.
func layerNorm(out, x []float32, weight, bias weightTensor, eps float64) {
	// Float64 reduction avoids cancellation in the variance. This computes the
	// same LayerNorm formula as the CPU float32 reference, without attempting
	// to reproduce its hardware-specific variance reduction order bit for bit.
	var mean, variance float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	inv := float32(1 / math.Sqrt(variance/float64(len(x))+eps))
	for i, v := range x {
		out[i] = (v-float32(mean))*inv*weight.at(i) + bias.at(i)
	}
}

// stableRotate uses split-half (NeoX) rotation on only the rotary prefix of
// each head. Llama's adjacent-pair/full-head rotation is not compatible.
func stableRotate(x []float32, headSize, rotary, pos int, theta float64) {
	for i := 0; i < rotary/2; i++ {
		// The reference materializes inverse frequencies and position products
		// as float32 tensors. Keeping the angle in float64 instead creates a
		// different phase at large positions even with more accurate trig.
		power := float32(math.Pow(float64(float32(theta)), float64(float32(2*i)/float32(rotary))))
		invFreq := float32(1) / power
		angle := float32(pos) * invFreq
		sin, cos := math.Sincos(float64(angle))
		sn, cs := float32(sin), float32(cos)
		for h := 0; h < len(x); h += headSize {
			a, b := x[h+i], x[h+i+rotary/2]
			// PyTorch evaluates the products as separate tensor operations.
			x[h+i] = float32(a*cs) - float32(b*sn)
			x[h+i+rotary/2] = float32(b*cs) + float32(a*sn)
		}
	}
}

// forward evaluates one token at its absolute position and writes KV into s.
// When logits is true it also fills s.logits. The caller owns s exclusively
// and has allocated enough capacity; a position becomes reusable only after
// the entire forward succeeds.
func (m *StableLM) forward(ctx context.Context, token, pos int, s *stableState, logits bool) error {
	c := m.config
	d := c.HiddenSize
	head := d / c.NumAttentionHeads
	rotary := int(float64(head) * c.PartialRotaryFactor)
	phase := s.profile.start()
	m.embed.readValues(s.x, token*d)
	phase = s.profile.mark("embedding", phase)
	for l, w := range m.layers {
		if err := ctx.Err(); err != nil {
			return err
		}
		layerNorm(s.norm, s.x, w.norm, w.bias, c.LayerNormEps)
		phase = s.profile.mark("layer_norm", phase)
		w.q.mulScratch(s.q, s.norm, s.matvecScratch)
		w.k.mulScratch(s.k, s.norm, s.matvecScratch)
		w.v.mulScratch(s.v, s.norm, s.matvecScratch)
		phase = s.profile.mark("qkv_projections", phase)
		stableRotate(s.q, head, rotary, pos, c.RopeTheta)
		stableRotate(s.k, head, rotary, pos, c.RopeTheta)
		phase = s.profile.mark("rotary", phase)
		// [KV head, position, channel] keeps each head's history contiguous
		// during attention. Only the new position is scattered across heads.
		capacity := len(s.att) / c.NumAttentionHeads
		for h := 0; h < c.NumKeyValueHeads; h++ {
			offset := (h*capacity + pos) * head
			s.storeKV(l, offset, s.k[h*head:(h+1)*head], s.v[h*head:(h+1)*head])
		}
		phase = s.profile.mark("kv_write", phase)
		m.prepareKV(l, pos+1, s)
		phase = s.profile.mark("kv_decode", phase)
		m.attend(l, pos, s)
		phase = s.profile.mark("attention", phase)
		w.o.mulScratch(s.proj, s.attOut, s.matvecScratch)
		phase = s.profile.mark("output_projection", phase)
		// Both branches consume the same input LayerNorm. Do not normalize the
		// attention residual a second time as in the Llama backend.
		w.gate.mulScratch(s.gate, s.norm, s.matvecScratch)
		w.up.mulScratch(s.up, s.norm, s.matvecScratch)
		for i, v := range s.gate {
			s.gate[i] = (v / (1 + float32(math.Exp(-float64(v))))) * s.up[i]
		}
		w.down.mulScratch(s.attOut, s.gate, s.matvecScratch)
		for i := range s.x {
			s.x[i] = (s.x[i] + s.proj[i]) + s.attOut[i]
		}
		phase = s.profile.mark("mlp", phase)
	}
	if logits {
		layerNorm(s.norm, s.x, m.norm, m.bias, c.LayerNormEps)
		phase = s.profile.mark("layer_norm", phase)
		m.head.mulScratch(s.logits, s.norm, s.matvecScratch)
		s.profile.mark("lm_head", phase)
	}
	return ctx.Err()
}

// attend gives each head its own score scratch and output slice. Splitting
// heads across workers does not change the reduction order within any head.
func (m *StableLM) attend(l, pos int, s *stableState) {
	parallelRows(m.config.NumAttentionHeads, m.config.HiddenSize*(pos+1), func(start, end int) {
		m.attendHeads(l, pos, s, start, end)
	})
}

// attendHeads computes causal attention for query heads in [start, end),
// including position pos. Each head uses disjoint score/output slices and
// reads its grouped KV head. The caller must first make the layer cache
// available through prepareKV.
func (m *StableLM) attendHeads(l, pos int, s *stableState, start, end int) {
	c := m.config
	head := c.HiddenSize / c.NumAttentionHeads
	kvGroup := c.NumAttentionHeads / c.NumKeyValueHeads
	capacity := len(s.att) / c.NumAttentionHeads
	scale := attentionScale(head)
	layerKeys, layerValues := s.kvLayer(l)
	for h := start; h < end; h++ {
		att := s.att[h*capacity : h*capacity+pos+1]
		q := s.q[h*head : (h+1)*head]
		offset := (h / kvGroup) * capacity * head
		keys := layerKeys[offset : offset+(pos+1)*head]
		values := layerValues[offset : offset+(pos+1)*head]
		for t := 0; t <= pos; t++ {
			k := keys[t*head : (t+1)*head]
			score := attentionDot(q, k)
			att[t] = score / scale
		}
		softmax(att)
		out := s.attOut[h*head : (h+1)*head]
		clear(out)
		for t, score := range att {
			v := values[t*head : (t+1)*head]
			attentionAdd(out, v, score)
		}
	}
}

// attentionScale returns the square root of the head width, the denominator
// used to scale attention logits before softmax.
func attentionScale(head int) float32 { return float32(math.Sqrt(float64(head))) }

// Generate continues plain text; Clio has no chat template or role training.
// One KV cache is retained across calls and grows in powers of two up to the
// model context limit. Matching token prefixes skip already-computed positions.
// EOS and matched stop strings are omitted from the returned/streamed text.
// Prompt usage includes BOS, and completion usage includes a sampled EOS.
//
// Requests wait cancellably for exclusive cache access. Completed prefill
// batches survive cancellation. Callbacks run synchronously while the request
// owns the cache and must not reenter this model. CPU KV precision is selected
// at load time with StableLMOptions; per-request KVBits and logprobs are rejected.
func (m *StableLM) Generate(prompt string, opts GenerateOptions) (GenerateResult, error) {
	if err := validateGeneration(opts); err != nil {
		return GenerateResult{}, err
	}
	if opts.Logprobs != 0 || opts.PromptLogprobs {
		return GenerateResult{}, fmt.Errorf("logprobs require native MLX")
	}
	if opts.KVBits != 0 {
		return GenerateResult{}, fmt.Errorf("kv_bits requires native MLX")
	}
	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return GenerateResult{}, err
	}
	window, err := contextLength(m.config.MaxPositionEmbeddings, opts.ContextLength)
	if err != nil {
		return GenerateResult{}, err
	}
	if err := validateContextTokenizer(opts, m.tokenizer.contextID); err != nil {
		return GenerateResult{}, err
	}
	ids, dropped, err := preparePrompt(prompt, opts, window, m.Encode)
	if err != nil {
		return GenerateResult{}, err
	}
	if len(ids) > window {
		return GenerateResult{}, fmt.Errorf("prompt has %d tokens (including BOS), exceeding context length %d", len(ids), window)
	}
	steps := window
	if opts.MaxTokens > 0 {
		steps = len(ids) + min(opts.MaxTokens-1, steps-len(ids))
	}
	var saved *stableContext
	select {
	case saved = <-m.cache:
	case <-ctx.Done():
		return GenerateResult{}, ctx.Err()
	}
	defer func() { m.cache <- saved }()
	// Cancellation and an available cache can become ready together. Check
	// again before a cancelled waiter grows or truncates a valid prefix.
	if err := ctx.Err(); err != nil {
		return GenerateResult{}, err
	}
	if saved == nil {
		saved = &stableContext{}
	}
	if saved.state != nil && len(saved.state.att)/m.config.NumAttentionHeads > window {
		saved = &stableContext{}
	}
	saved.resize(m, steps)
	s := saved.state
	prefix := reusablePrefix(saved.tokens, ids)
	// Re-evaluate the final prompt token even on a full hit to obtain its
	// logits. A failed/cancelled suffix never becomes valid cached state.
	saved.tokens = saved.tokens[:prefix]
	s.prefillDone = func(end int) { saved.tokens = append(saved.tokens, ids[len(saved.tokens):end]...) }
	err = m.prefill(ctx, ids[prefix:], prefix, s)
	s.prefillDone = nil
	if err != nil {
		return GenerateResult{}, err
	}
	rng := rand.New(rand.NewSource(opts.Seed))
	result := GenerateResult{PromptTokens: len(ids), CachedPromptTokens: prefix, FinishReason: "length"}
	result.TruncatedPromptTokens = dropped
	history := append([]int(nil), ids...)
	output := completionOutput{stops: opts.Stop, onToken: opts.OnToken}
	decoder := nerdstashDecoder{tokenizer: m.tokenizer}
	for pos := len(ids) - 1; pos < steps; pos++ {
		if err := ctx.Err(); err != nil {
			return GenerateResult{}, err
		}
		token := int(sampleWithOptions(s.logits, history, opts, rng))
		history = append(history, token)
		result.CompletionTokens++
		if token == m.config.EOSTokenID {
			result.FinishReason = "stop"
			break
		}
		if output.add(decoder.add(token), false) {
			result.FinishReason = "stop"
			break
		}
		if pos+1 < steps {
			if err := m.forward(ctx, token, pos+1, s, true); err != nil {
				return GenerateResult{}, err
			}
			saved.tokens = append(saved.tokens, token)
		}
	}
	if output.add(decoder.flush(), true) {
		result.FinishReason = "stop"
	}
	result.Text = output.text.String()
	return result, nil
}

// completionOutput holds potential stop prefixes back from streaming clients.
type completionOutput struct {
	text    strings.Builder
	pending string
	stops   []string
	onToken func(string)
	stopped bool
}

// add appends decoded text, emits only text that cannot become part of a stop
// string, and reports whether a stop was found. Setting final releases an
// unmatched stop prefix at end of generation. Once stopped, further calls emit
// nothing.
func (o *completionOutput) add(piece string, final bool) bool {
	if o.stopped {
		return true
	}
	o.pending += piece
	end := len(o.pending)
	for _, stop := range o.stops {
		if stop == "" {
			continue
		}
		if i := strings.Index(o.pending, stop); i >= 0 && i < end {
			end = i
			o.stopped = true
		}
	}
	if !o.stopped && !final {
		for _, stop := range o.stops {
			for n := 1; n < len(stop) && n <= len(o.pending); n++ {
				if strings.HasSuffix(o.pending, stop[:n]) {
					end = min(end, len(o.pending)-n)
				}
			}
		}
	}
	if end > 0 {
		text := o.pending[:end]
		o.text.WriteString(text)
		if o.onToken != nil {
			o.onToken(text)
		}
	}
	o.pending = o.pending[end:]
	return o.stopped
}
