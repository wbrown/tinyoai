package tinyoai

// This file vendors a pure-Go Llama 2 inference engine adapted from
// github.com/tmc/go-llama2 (MIT), itself a port of Andrej Karpathy's
// llama2.c (MIT, https://github.com/karpathy/llama2.c). It reads the legacy
// llama2.c checkpoint format (a 7-int32 Config header followed by raw
// little-endian float32 weights, classifier tied to the token embedding).
//
// Two corrections were made versus the upstream port:
//   - The rotary (RoPE) tables freq_cis_real/imag are sized seqLen*headSize/2,
//     matching the on-disk layout, not the upstream seqLen*dim/2 which over-reads
//     and fails to load them.
//   - The per-timestep attention score is written to the attention buffer before
//     the softmax; upstream computed it and discarded it.
//
// The engine is pure standard library: no CGO, no external dependencies.

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"strings"
)

// Config is the model hyper-parameter header at the start of a checkpoint.
type Config struct {
	Dim       int32 // transformer dimension
	HiddenDim int32 // FFN hidden dimension
	NLayers   int32 // number of layers
	NHeads    int32 // number of attention heads
	NKvHeads  int32 // number of key/value heads (can be fewer for grouped-query attention)
	VocabSize int32 // vocabulary size
	SeqLen    int32 // maximum sequence length
}

// transformerWeights holds the model parameters.
type transformerWeights struct {
	tokenEmbeddingTable []float32 // (vocab, dim)
	rmsAttWeight        []float32 // (layer, dim)
	rmsFfnWeight        []float32 // (layer, dim)
	wq                  []float32 // (layer, dim, dim)
	wk                  []float32 // (layer, kv_dim, dim)
	wv                  []float32 // (layer, kv_dim, dim)
	wo                  []float32 // (layer, dim, dim)
	w1                  []float32 // (layer, hidden, dim)
	w2                  []float32 // (layer, dim, hidden)
	w3                  []float32 // (layer, hidden, dim)
	rmsFinalWeight      []float32 // (dim,)
	freqCisReal         []float32 // (seq_len, head_size/2)
	freqCisImag         []float32 // (seq_len, head_size/2)
}

// runState holds the per-generation activation buffers.
type runState struct {
	x          []float32
	xb         []float32
	xb2        []float32
	hb         []float32
	hb2        []float32
	q          []float32
	k          []float32
	v          []float32
	att        []float32
	logits     []float32
	keyCache   []float32
	valueCache []float32
}

// Model is a loaded checkpoint plus tokenizer, ready for generation. Its weights
// and vocabulary are read-only after loading, so a single Model is safe for
// concurrent Generate calls (each call allocates its own run state).
type Model struct {
	config         Config
	weights        transformerWeights
	vocab          []string
	vocabScores    []float32
	maxTokenLength uint32
}

// Config returns the model's hyper-parameters.
func (m *Model) Config() Config { return m.config }

// LoadModel reads a legacy llama2.c checkpoint and matching tokenizer from
// caller-owned streams. It validates basic dimensions and exact checkpoint
// consumption; truncation and unsupported layouts return errors. Inputs must
// be trusted local model files: allocations follow their declared dimensions.
func LoadModel(checkpoint, tokenizer io.Reader) (*Model, error) {
	var config Config
	if err := binary.Read(checkpoint, binary.LittleEndian, &config); err != nil {
		return nil, fmt.Errorf("read config header: %w", err)
	}
	if config.Dim <= 0 || config.HiddenDim <= 0 || config.NLayers <= 0 || config.NHeads <= 0 ||
		config.NKvHeads <= 0 || config.NHeads%config.NKvHeads != 0 ||
		config.VocabSize < 2 || config.SeqLen <= 0 || config.Dim%config.NHeads != 0 || (config.Dim/config.NHeads)%2 != 0 {
		return nil, fmt.Errorf("invalid config header: %+v", config)
	}

	weights, err := readWeights(checkpoint, &config)
	if err != nil {
		return nil, fmt.Errorf("read weights: %w", err)
	}
	// The whole checkpoint must be consumed; trailing bytes mean the layout
	// does not match the file, which would silently corrupt inference.
	if n, _ := io.Copy(io.Discard, checkpoint); n != 0 {
		return nil, fmt.Errorf("checkpoint has %d trailing bytes: weight layout does not match file", n)
	}

	vocab, scores, maxLen, err := readTokenizer(tokenizer, config.VocabSize)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer: %w", err)
	}

	return &Model{
		config:         config,
		weights:        weights,
		vocab:          vocab,
		vocabScores:    scores,
		maxTokenLength: maxLen,
	}, nil
}

// readWeights reads row-major tensors in the legacy checkpoint order. Query
// projections use all attention heads; key/value projections use only the
// grouped KV heads. The classifier reuses the embedding table, so no separate
// output matrix is read.
func readWeights(r io.Reader, p *Config) (transformerWeights, error) {
	headSize := int(p.Dim / p.NHeads)
	kvDim := headSize * int(p.NKvHeads) // grouped-query attention: kv heads <= query heads
	dim := int(p.Dim)
	hidden := int(p.HiddenDim)
	layers := int(p.NLayers)
	vocab := int(p.VocabSize)
	seq := int(p.SeqLen)

	var w transformerWeights
	// (slice, element-count) pairs in the order they appear on disk.
	tensors := []struct {
		dst  *[]float32
		size int
	}{
		{&w.tokenEmbeddingTable, vocab * dim},
		{&w.rmsAttWeight, layers * dim},
		{&w.wq, layers * dim * dim},
		{&w.wk, layers * dim * kvDim},
		{&w.wv, layers * dim * kvDim},
		{&w.wo, layers * dim * dim},
		{&w.rmsFfnWeight, layers * dim},
		{&w.w1, layers * hidden * dim},
		{&w.w2, layers * dim * hidden},
		{&w.w3, layers * hidden * dim},
		{&w.rmsFinalWeight, dim},
		{&w.freqCisReal, seq * headSize / 2},
		{&w.freqCisImag, seq * headSize / 2},
	}
	for _, t := range tensors {
		buf := make([]float32, t.size)
		if err := binary.Read(r, binary.LittleEndian, buf); err != nil {
			return w, err
		}
		*t.dst = buf
	}
	return w, nil
}

// readTokenizer reads the llama2.c tokenizer.bin format: a uint32 maximum
// token length followed by vocabSize entries of float32 score, int32 byte
// length, and text. Scores rank candidate BPE merges rather than output
// probabilities.
func readTokenizer(r io.Reader, vocabSize int32) ([]string, []float32, uint32, error) {
	var maxTokenLength uint32
	if err := binary.Read(r, binary.LittleEndian, &maxTokenLength); err != nil {
		return nil, nil, 0, fmt.Errorf("read max token length: %w", err)
	}
	// Legacy tokenizers are small text vocabularies. Bound both the declared
	// maximum and each entry before allocating from checkpoint metadata.
	if maxTokenLength == 0 || maxTokenLength > 1<<20 {
		return nil, nil, 0, fmt.Errorf("invalid maximum token length %d", maxTokenLength)
	}
	vocab := make([]string, vocabSize)
	scores := make([]float32, vocabSize)
	for i := int32(0); i < vocabSize; i++ {
		if err := binary.Read(r, binary.LittleEndian, &scores[i]); err != nil {
			return nil, nil, 0, fmt.Errorf("read score %d: %w", i, err)
		}
		var length int32
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return nil, nil, 0, fmt.Errorf("read length %d: %w", i, err)
		}
		if length < 0 || uint32(length) > maxTokenLength {
			return nil, nil, 0, fmt.Errorf("invalid token %d length %d (maximum %d)", i, length, maxTokenLength)
		}
		b := make([]byte, length)
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, nil, 0, fmt.Errorf("read token %d bytes: %w", i, err)
		}
		vocab[i] = string(b)
	}
	return vocab, scores, maxTokenLength, nil
}

// GenerateOptions configures one generation call. Fields for prepared prompts
// and truncation apply to StableLM backends; probability capture requires native
// MLX. Legacy Model ignores the prepared-prompt fields. Callbacks are synchronous
// and must not reenter a backend that holds an exclusive request gate.
type GenerateOptions struct {
	// ExpectedPromptTokens checks a positive client-prepared token count before
	// inference. StableLM counts BOS; LlamaMLX excludes it. Zero disables checking.
	ExpectedPromptTokens int
	// TokenizerID optionally checks a StableLM prepared prompt against
	// ContextTokenizer().ID().
	TokenizerID string
	// Logprobs retains this many raw-model alternatives per token (native MLX).
	Logprobs int
	// PromptLogprobs also captures known prompt-token scores; it requires
	// positive Logprobs and native MLX.
	PromptLogprobs bool
	// OnLogprobs receives synchronous raw-model probability events when capture
	// is enabled.
	OnLogprobs func(ProbabilityEvent)
	// ContextLength optionally uses a smaller window than the model supports.
	ContextLength int
	// KVBits selects backend KV storage precision; 0 uses that backend's default.
	KVBits int
	// OnProgress reports completed MLX forwards, after GPU evaluation finishes.
	OnProgress func(phase string, cachedTokens int)
	// Sampling enables the extended sampler when non-nil; nil preserves the
	// legacy temperature-only sampler.
	Sampling *SamplingOptions
	// TruncatePrompt keeps BOS and the most recent context, reserving output space.
	TruncatePrompt bool
	// PromptPrefix, when present, must begin prompt and is retained on truncation.
	PromptPrefix string
	// Context optionally cancels inference when a request is abandoned.
	Context     context.Context
	MaxTokens   int     // Maximum completion tokens; zero uses remaining context.
	Temperature float64 // 0 = greedy argmax
	Seed        int64   // RNG seed for reproducible sampling
	// Stop lists stop strings. StableLM withholds matches; legacy Llama emits
	// the matching text before stopping.
	Stop []string
	// OnToken streams decoded completion text. A callback need not correspond
	// to exactly one token: UTF-8 and partial stop markers may be buffered.
	OnToken func(piece string)
}

// GenerateResult is the outcome of one generation call.
type GenerateResult struct {
	// Logprobs records completion-token probabilities when capture is enabled.
	Logprobs []TokenLogprob
	// PromptLogprobs records scores for known prompt tokens when requested.
	PromptLogprobs []TokenLogprob
	// TruncatedPromptTokens counts tokens removed by backend prompt truncation.
	TruncatedPromptTokens int
	// Text is the completion after backend decoding and stop handling.
	Text string
	// PromptTokens counts prompt tokens; StableLM includes BOS, while legacy
	// Llama excludes it.
	PromptTokens       int
	CachedPromptTokens int // prompt tokens reused by a backend with prefix caching
	// CompletionTokens counts sampled tokens; StableLM includes EOS, while
	// legacy Llama excludes its BOS stop sentinel.
	CompletionTokens int
	FinishReason     string // "stop", "length", or "prefill" for Prefill.
}

// Generate runs autoregressive decoding with private activation and KV
// buffers, allowing concurrent calls on the same immutable Model. It stops at
// the BOS sentinel, a stop-string match, MaxTokens, or the context boundary.
//
// Legacy Llama accounting excludes BOS from prompt usage and does not count
// the stop sentinel as a completion. Stop text is emitted before matching and
// remains in the result. Callbacks are synchronous; no prefix survives between
// calls.
func (m *Model) Generate(prompt string, opts GenerateOptions) (GenerateResult, error) {
	if err := validateGeneration(opts); err != nil {
		return GenerateResult{}, err
	}
	if opts.Logprobs != 0 || opts.PromptLogprobs {
		return GenerateResult{}, fmt.Errorf("logprobs require native MLX")
	}
	if opts.KVBits != 0 {
		return GenerateResult{}, fmt.Errorf("kv_bits requires native MLX")
	}
	promptTokens, err := m.bpeEncode(prompt)
	if err != nil {
		return GenerateResult{}, err
	}

	maxSteps, err := contextLength(int(m.config.SeqLen), opts.ContextLength)
	if err != nil {
		return GenerateResult{}, err
	}
	state := m.newRunState()
	rng := rand.New(rand.NewSource(opts.Seed))

	var out strings.Builder
	completion := 0
	finish := "length"
	history := append([]int{1}, promptTokens...)
	token := int32(1) // BOS, per the Llama-2 sentencepiece tokenizer

	for pos := 0; pos < maxSteps; pos++ {
		if opts.Context != nil {
			if err := opts.Context.Err(); err != nil {
				return GenerateResult{}, err
			}
		}
		m.transformer(token, int32(pos), state)

		var next int32
		if pos < len(promptTokens) {
			next = int32(promptTokens[pos])
		} else {
			next = sampleWithOptions(state.logits, history, opts, rng)
			history = append(history, int(next))
			if next == 1 { // BOS/sentinel marks an end of story
				finish = "stop"
				break
			}
			piece := m.tokenPiece(token, next)
			out.WriteString(piece)
			if opts.OnToken != nil {
				opts.OnToken(piece)
			}
			completion++
			if matchesStop(out.String(), opts.Stop) {
				finish = "stop"
				break
			}
			if opts.MaxTokens > 0 && completion >= opts.MaxTokens {
				finish = "length"
				break
			}
		}
		token = next
	}

	return GenerateResult{
		Text:             out.String(),
		PromptTokens:     len(promptTokens),
		CompletionTokens: completion,
		FinishReason:     finish,
	}, nil
}

// tokenPiece renders a token's text, trimming a leading space immediately after
// the BOS token (the sentencepiece convention).
func (m *Model) tokenPiece(prevToken, next int32) string {
	piece := m.vocab[next]
	if prevToken == 1 && strings.HasPrefix(piece, " ") {
		piece = piece[1:]
	}
	return piece
}

// matchesStop reports whether text contains any nonempty stop string. The tiny
// Llama backend checks after emitting a piece, so it includes the matching
// stop text in its output.
func matchesStop(text string, stops []string) bool {
	for _, s := range stops {
		if s != "" && strings.Contains(text, s) {
			return true
		}
	}
	return false
}

// newRunState allocates independent activations and full-context KV for one
// legacy Llama request. KV is position-major within each layer, unlike
// StableLM's head-major storage.
func (m *Model) newRunState() *runState {
	p := &m.config
	dim := p.Dim
	kvDim := (p.Dim / p.NHeads) * p.NKvHeads
	return &runState{
		x:          make([]float32, dim),
		xb:         make([]float32, dim),
		xb2:        make([]float32, dim),
		hb:         make([]float32, p.HiddenDim),
		hb2:        make([]float32, p.HiddenDim),
		q:          make([]float32, dim),
		k:          make([]float32, kvDim),
		v:          make([]float32, kvDim),
		att:        make([]float32, p.SeqLen),
		logits:     make([]float32, p.VocabSize),
		keyCache:   make([]float32, p.NLayers*p.SeqLen*kvDim),
		valueCache: make([]float32, p.NLayers*p.SeqLen*kvDim),
	}
}

// transformer evaluates token at absolute position pos and replaces s.logits
// with the next-token distribution. Earlier cache positions must already be
// populated.
//
// Each layer first adds its attention output to the residual, then normalizes
// that updated residual for the gated MLP. Grouped query heads share KV;
// adjacent coordinate pairs use checkpoint rotary tables. The final projection
// reuses the embedding matrix.
func (m *Model) transformer(token, pos int32, s *runState) {
	w := &m.weights
	p := &m.config
	x := s.x
	dim := p.Dim
	hiddenDim := p.HiddenDim
	headSize := dim / p.NHeads
	kvDim := headSize * p.NKvHeads // total key/value width
	kvMul := p.NHeads / p.NKvHeads // query heads sharing each kv head

	copy(x, w.tokenEmbeddingTable[token*dim:(token+1)*dim])

	freqCisRealRow := w.freqCisReal[pos*headSize/2 : (pos+1)*headSize/2]
	freqCisImagRow := w.freqCisImag[pos*headSize/2 : (pos+1)*headSize/2]

	for l := int32(0); l < p.NLayers; l++ {
		rmsNorm(s.xb, x, w.rmsAttWeight[l*dim:(l+1)*dim])

		matmul(s.q, s.xb, w.wq[l*dim*dim:(l+1)*dim*dim])
		matmul(s.k, s.xb, w.wk[l*dim*kvDim:(l+1)*dim*kvDim])
		matmul(s.v, s.xb, w.wv[l*dim*kvDim:(l+1)*dim*kvDim])

		// RoPE: rotate each query head, then each (fewer) key head.
		for h := int32(0); h < p.NHeads; h++ {
			rotateHead(s.q[h*headSize:(h+1)*headSize], freqCisRealRow, freqCisImagRow, headSize)
		}
		for h := int32(0); h < p.NKvHeads; h++ {
			rotateHead(s.k[h*headSize:(h+1)*headSize], freqCisRealRow, freqCisImagRow, headSize)
		}

		// Cache key/value at this position.
		loff := l * p.SeqLen * kvDim
		copy(s.keyCache[loff+pos*kvDim:loff+(pos+1)*kvDim], s.k)
		copy(s.valueCache[loff+pos*kvDim:loff+(pos+1)*kvDim], s.v)

		// Multi-head attention; each query head reads its grouped kv head.
		for h := int32(0); h < p.NHeads; h++ {
			q := s.q[h*headSize : (h+1)*headSize]
			kvOff := (h / kvMul) * headSize
			for t := int32(0); t <= pos; t++ {
				k := s.keyCache[loff+t*kvDim+kvOff : loff+t*kvDim+kvOff+headSize]
				score := float32(0)
				for i := int32(0); i < headSize; i++ {
					score += q[i] * k[i]
				}
				s.att[t] = score / float32(math.Sqrt(float64(headSize)))
			}
			softmax(s.att[:pos+1])
			for i := int32(0); i < headSize; i++ {
				val := float32(0)
				for t := int32(0); t <= pos; t++ {
					val += s.att[t] * s.valueCache[loff+t*kvDim+kvOff+i]
				}
				s.xb[h*headSize+i] = val
			}
		}

		matmul(s.xb2, s.xb, w.wo[l*dim*dim:(l+1)*dim*dim])
		accum(x, s.xb2)

		// FFN: w2(silu(w1(x)) * w3(x)).
		rmsNorm(s.xb, x, w.rmsFfnWeight[l*dim:(l+1)*dim])
		matmul(s.hb, s.xb, w.w1[l*dim*hiddenDim:(l+1)*dim*hiddenDim])
		matmul(s.hb2, s.xb, w.w3[l*dim*hiddenDim:(l+1)*dim*hiddenDim])
		for i := int32(0); i < hiddenDim; i++ {
			s.hb[i] = s.hb[i] * (1.0 / (1.0 + float32(math.Exp(-float64(s.hb[i])))))
			s.hb[i] = s.hb[i] * s.hb2[i]
		}
		matmul(s.xb, s.hb, w.w2[l*dim*hiddenDim:(l+1)*dim*hiddenDim])
		accum(x, s.xb)
	}

	rmsNorm(x, x, w.rmsFinalWeight)
	matmul(s.logits, x, w.tokenEmbeddingTable) // classifier tied to embedding
}

// sampleNext returns a greedy argmax at temperature zero, or samples
// temperature-scaled softmax probabilities with rng. The sampling path
// overwrites logits; callers needing raw model scores must copy them first.
func sampleNext(logits []float32, temperature float64, rng *rand.Rand) int32 {
	if temperature == 0 {
		return argmax(logits)
	}
	for i := range logits {
		logits[i] /= float32(temperature)
	}
	softmax(logits)
	return sampleDist(logits, rng)
}

// ----------------------------------------------------------------------------
// Tokenizer (byte-pair encoding)

// bpeEncode encodes text into token ids, greedily merging the highest-scoring
// adjacent pair each pass.
func (m *Model) bpeEncode(text string) ([]int, error) {
	tokens := make([]int, 0, len(text))
	for _, c := range text {
		id := m.strLookup(string(c))
		if id < 0 {
			return nil, fmt.Errorf("tokenizer: character %q not in vocabulary", string(c))
		}
		tokens = append(tokens, id)
	}

	var buf strings.Builder
	for {
		bestScore := float32(-1e10)
		bestID, bestIdx := -1, -1
		for i := 0; i < len(tokens)-1; i++ {
			buf.Reset()
			buf.WriteString(m.vocab[tokens[i]])
			buf.WriteString(m.vocab[tokens[i+1]])
			id := m.strLookup(buf.String())
			if id >= 0 && m.vocabScores[id] > bestScore {
				bestScore = m.vocabScores[id]
				bestID = id
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		tokens[bestIdx] = bestID
		tokens = append(tokens[:bestIdx+1], tokens[bestIdx+2:]...)
	}
	return tokens, nil
}

// strLookup returns the vocab index of str, or -1 if absent.
func (m *Model) strLookup(str string) int {
	for i, v := range m.vocab {
		if v == str {
			return i
		}
	}
	return -1
}

// ----------------------------------------------------------------------------
// Math primitives

// softmax replaces logits with normalized probabilities in place. Subtracting
// the largest logit prevents exponential overflow; an empty slice is
// unchanged. Inputs are expected to be finite.
func softmax(x []float32) {
	if len(x) == 0 {
		return
	}
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	for i, v := range x {
		x[i] = float32(math.Exp(float64(v - maxVal)))
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
}

// matmul writes W*x to xout, interpreting w as contiguous output rows of
// len(x) elements. This small-model reference uses sequential float32
// accumulation. The input must be nonempty and must not overlap the output.
func matmul(xout, x, w []float32) {
	n := len(x)
	d := len(w) / n
	for i := 0; i < d; i++ {
		var val float32
		row := w[i*n : i*n+n]
		for j := 0; j < n; j++ {
			val += row[j] * x[j]
		}
		xout[i] = val
	}
}

// rmsNorm writes weight-scaled root-mean-square normalization of src to dest,
// using the legacy Llama epsilon of 1e-5. All slices have the same nonzero
// length; dest may alias src.
func rmsNorm(dest, src, weight []float32) {
	var sumSquares float32
	for _, v := range src {
		sumSquares += v * v
	}
	ss := 1.0 / float32(math.Sqrt(float64(sumSquares/float32(len(src))+1e-5)))
	for i, v := range src {
		dest[i] = weight[i] * (ss * v)
	}
}

// accum adds b to a in place. Both vectors must have the same length.
func accum(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// rotateHead applies the precomputed RoPE rotation (one row of freq_cis) to a
// single head's vector in place.
func rotateHead(vec, freqReal, freqImag []float32, headSize int32) {
	for i := int32(0); i < headSize; i += 2 {
		v0, v1 := vec[i], vec[i+1]
		fcr, fci := freqReal[i/2], freqImag[i/2]
		vec[i] = v0*fcr - v1*fci
		vec[i+1] = v0*fci + v1*fcr
	}
}

// argmax returns the index of the largest element of a nonempty slice. Strict
// comparison preserves the earliest index when logits tie.
func argmax(v []float32) int32 {
	maxI, maxP := 0, v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > maxP {
			maxI, maxP = i, v[i]
		}
	}
	return int32(maxI)
}

// sampleDist draws an index using rng from a nonempty probability distribution
// whose sum is approximately one. If rounding leaves a gap at the end of the
// cumulative distribution, the last index absorbs it.
func sampleDist(probabilities []float32, rng *rand.Rand) int32 {
	r := rng.Float32()
	var cdf float32
	for i, p := range probabilities {
		cdf += p
		if r < cdf {
			return int32(i)
		}
	}
	return int32(len(probabilities) - 1)
}
