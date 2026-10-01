package tinyoai

import (
	"bufio"
	"context"
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

//go:embed scripts/mlx_worker.py
var mlxWorker string

// MLX uses a native MLX C backend or a persistent Python MLX-LM worker for
// forward passes. Tokenization, sampling, stop handling and the HTTP API are
// shared with the Go CPU backend.
type MLX struct {
	python, dir string
	config      StableLMConfig
	tokenizer   *nerdstashTokenizer
	gate        chan struct{}
	cmd         *exec.Cmd
	in          io.WriteCloser
	out         *bufio.Reader
	tokens      []int
	probRows    []logprobRow
	probCount   int
	closed      bool
	native      mlxForwarder
	// Zero keeps the production default. Experiments change this only while idle.
	prefillChunk int
}

type mlxForwarder interface {
	// Forward evaluates tokens after a resident prefix and optionally returns
	// the final vocabulary row. Completed chunks remain reusable on cancellation.
	Forward(context.Context, int, []int, bool) ([]float32, error)
	// Close releases native resources after callers stop using the forwarder.
	Close() error
}

type mlxProbabilityForwarder interface {
	// ForwardWithLogprobs pairs every input position with a target and returns
	// complete chosen/top probability rows plus the final full-vocabulary logits.
	ForwardWithLogprobs(context.Context, int, []int, []int, int) ([]float32, []logprobRow, error)
}

type mlxRequest struct {
	Prefix int   `json:"prefix"`
	Tokens []int `json:"tokens"`
	Logits bool  `json:"logits"`
}

// LoadMLX loads a StableLM-compatible checkpoint through a persistent Python
// MLX-LM worker using the given interpreter. Startup has a two-minute
// deadline. The returned model owns the worker and must be closed.
func LoadMLX(python, dir string) (*MLX, error) {
	m, err := newMLX(dir)
	if err != nil {
		return nil, err
	}
	m.python = python
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if err = m.start(ctx); err != nil {
		return nil, err
	}
	return m, nil
}

// newMLX validates configuration and tokenizer data and initializes the
// exclusive request gate. It does not load weights or start an inference
// backend.
func newMLX(dir string) (*MLX, error) {
	m := &MLX{dir: dir, gate: make(chan struct{}, 1)}
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	if err = json.Unmarshal(b, &m.config); err != nil {
		return nil, err
	}
	if err = m.config.validate(); err != nil {
		return nil, err
	}
	m.tokenizer, err = loadNerdstash(filepath.Join(dir, "tokenizer.json"), m.config.VocabSize, m.config.BOSTokenID)
	if err != nil {
		return nil, err
	}
	m.gate <- struct{}{}
	return m, nil
}

// start launches the embedded Python worker and waits for its readiness reply.
// The caller holds the model gate; exchange kills a failed or cancelled
// worker.
func (m *MLX) start(ctx context.Context) error {
	m.cmd = exec.Command(m.python, "-u", "-c", mlxWorker, m.dir)
	m.cmd.Stderr = os.Stderr
	var err error
	m.in, err = m.cmd.StdinPipe()
	if err != nil {
		return err
	}
	out, err := m.cmd.StdoutPipe()
	if err != nil {
		_ = m.in.Close()
		return err
	}
	m.out = bufio.NewReaderSize(out, 1<<20)
	if err = m.cmd.Start(); err != nil {
		_ = m.in.Close()
		_ = out.Close()
		m.cmd = nil
		return err
	}
	_, err = m.exchange(ctx, nil)
	return err
}

// kill closes and reaps the Python worker, then invalidates Go's retained
// prefix and probability rows. The caller holds the model gate.
func (m *MLX) kill() {
	if m.cmd != nil {
		_ = m.in.Close()
		_ = m.cmd.Process.Kill()
		_ = m.cmd.Wait()
		m.cmd = nil
	}
	m.tokens = nil
	m.probRows = nil
}

// Close waits for the active request or branch session, then releases the
// worker or native backend and retained prefix. Repeated calls are harmless.
func (m *MLX) Close() error {
	<-m.gate
	defer func() { m.gate <- struct{}{} }()
	if m.closed {
		return nil
	}
	m.closed = true
	m.kill()
	if m.native != nil {
		return m.native.Close()
	}
	return nil
}

// exchange performs one forward chunk through the selected backend. A nil
// request is reserved for Python startup. The caller holds the model gate.
//
// Python replies are a JSON byte-count header followed by little-endian
// float32 logits. Cancellation kills and reaps the worker before returning, so
// unread response bytes cannot corrupt a subsequent request. Native
// cancellation retains completed chunks; other native errors invalidate the
// recorded prefix.
func (m *MLX) exchange(ctx context.Context, request *mlxRequest) ([]float32, error) {
	if m.native != nil {
		logits, err := m.native.Forward(ctx, request.Prefix, request.Tokens, request.Logits)
		if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			m.tokens = nil
		}
		return logits, err
	}
	type reply struct {
		values []float32
		err    error
	}
	done := make(chan reply, 1)
	go func() {
		if request != nil {
			if err := json.NewEncoder(m.in).Encode(request); err != nil {
				done <- reply{err: err}
				return
			}
		}
		line, err := m.out.ReadBytes('\n')
		if err != nil {
			done <- reply{err: fmt.Errorf("MLX worker: %w (see server log)", err)}
			return
		}
		var h struct {
			Bytes int    `json:"bytes"`
			Error string `json:"error"`
		}
		if err = json.Unmarshal(line, &h); err != nil {
			done <- reply{err: err}
			return
		}
		if h.Error != "" {
			done <- reply{err: fmt.Errorf("MLX: %s", h.Error)}
			return
		}
		if h.Bytes != 0 && h.Bytes != m.config.VocabSize*4 {
			done <- reply{err: fmt.Errorf("invalid MLX logits size %d", h.Bytes)}
			return
		}
		data := make([]byte, h.Bytes)
		if _, err = io.ReadFull(m.out, data); err != nil {
			done <- reply{err: err}
			return
		}
		values := make([]float32, h.Bytes/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		}
		done <- reply{values: values}
	}()
	select {
	case r := <-done:
		if r.err != nil {
			m.kill()
		}
		return r.values, r.err
	case <-ctx.Done():
		m.kill()
		<-done
		return nil, ctx.Err()
	}
}

// Generate completes a prompt using shared Go tokenization, sampling, and stop
// handling. Requests serialize on the model gate and reuse an unchanged
// resident prefix. Callbacks run synchronously under that gate and must not
// reenter the model.
func (m *MLX) Generate(prompt string, opts GenerateOptions) (GenerateResult, error) {
	return m.generate(prompt, opts, false)
}

// Prefill prepares the same KV and probability rows as Generate without sampling
// or emitting a completion. It uses the same gate and keeps completed chunks.
func (m *MLX) Prefill(prompt string, opts GenerateOptions) (GenerateResult, error) {
	return m.generate(prompt, opts, true)
}

// generate validates request policy, prepares the prompt, and owns the model
// gate across prefill and optional decoding. Cached token IDs and probability
// rows are committed only after a complete forward chunk.
//
// The final prompt token is evaluated again to recover next-token logits. When
// prompt probabilities are requested, one additional preceding row is
// recomputed because it predicts the first changed token. Context or KV
// precision changes may discard the whole prefix.
func (m *MLX) generate(prompt string, opts GenerateOptions, prefillOnly bool) (GenerateResult, error) {
	if err := validateGeneration(opts); err != nil {
		return GenerateResult{}, err
	}
	probNative, supportsProbs := m.native.(mlxProbabilityForwarder)
	if opts.Logprobs > 0 && !supportsProbs {
		return GenerateResult{}, fmt.Errorf("logprobs requires native MLX")
	}
	if opts.KVBits != 0 && opts.KVBits != 8 && opts.KVBits != 16 {
		return GenerateResult{}, fmt.Errorf("kv_bits must be 0 (FP16 default), 8, or 16")
	}
	if opts.KVBits != 0 && m.native == nil {
		return GenerateResult{}, fmt.Errorf("kv_bits requires native MLX")
	}
	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}
	window, err := contextLength(m.config.MaxPositionEmbeddings, opts.ContextLength)
	if err != nil {
		return GenerateResult{}, err
	}
	if err := validateContextTokenizer(opts, m.tokenizer.contextID); err != nil {
		return GenerateResult{}, err
	}
	ids, dropped, err := preparePrompt(prompt, opts, window, m.tokenizer.encode)
	if err != nil {
		return GenerateResult{}, err
	}
	if len(ids) > window {
		return GenerateResult{}, fmt.Errorf("prompt exceeds %d tokens", window)
	}
	select {
	case <-ctx.Done():
		return GenerateResult{}, ctx.Err()
	case <-m.gate:
	}
	defer func() { m.gate <- struct{}{} }()
	if err := ctx.Err(); err != nil {
		return GenerateResult{}, err
	}
	if m.closed {
		return GenerateResult{}, fmt.Errorf("MLX backend is closed")
	}
	if native, ok := m.native.(interface{ SetKVBits(int) (bool, error) }); ok {
		bits := opts.KVBits
		if bits == 0 {
			bits = 16
		}
		cleared, err := native.SetKVBits(bits)
		if err != nil {
			return GenerateResult{}, err
		}
		if cleared {
			m.tokens = nil
		}
	}
	if native, ok := m.native.(interface{ LimitCache(int) (bool, error) }); ok {
		cleared, err := native.LimitCache(window)
		if err != nil {
			return GenerateResult{}, err
		}
		if cleared {
			m.tokens = nil
		}
	}
	if m.cmd == nil && m.native == nil {
		if err := m.start(ctx); err != nil {
			return GenerateResult{}, err
		}
	}
	prefix := reusablePrefix(m.tokens, ids)
	prefix = prepareProbabilityCache(prefix, &m.probRows, &m.probCount, opts)
	m.tokens = m.tokens[:prefix]
	var logits []float32
	for pos := prefix; pos < len(ids); {
		if err := ctx.Err(); err != nil {
			return GenerateResult{}, err
		}
		chunk := m.prefillChunk
		if chunk == 0 {
			chunk = 512
		}
		end := min(pos+chunk, len(ids))
		var err error
		if opts.PromptLogprobs {
			targets := make([]int, end-pos)
			copy(targets, ids[pos+1:min(end+1, len(ids))])
			var rows []logprobRow
			logits, rows, err = probNative.ForwardWithLogprobs(ctx, pos, ids[pos:end], targets, opts.Logprobs)
			if err == nil {
				m.probRows = append(m.probRows, rows...)
			} else if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
				m.tokens = nil
				m.probRows = nil
			}
		} else {
			logits, err = m.exchange(ctx, &mlxRequest{pos, ids[pos:end], end == len(ids)})
		}
		if err != nil {
			return GenerateResult{}, err
		}
		m.tokens = append(m.tokens, ids[pos:end]...)
		pos = end
		if opts.OnProgress != nil {
			opts.OnProgress("prefill", pos)
		}
	}
	if err := ctx.Err(); err != nil {
		return GenerateResult{}, err
	}
	limit := window - len(ids) + 1
	if opts.MaxTokens > 0 {
		limit = min(limit, opts.MaxTokens)
	}
	result := GenerateResult{PromptTokens: len(ids), CachedPromptTokens: prefix, TruncatedPromptTokens: dropped, FinishReason: "length"}
	if opts.PromptLogprobs {
		fingerprint := extendFingerprint(14695981039346656037, ids[0])
		spans := m.tokenizer.promptSpans(prompt, opts.PromptPrefix, ids, dropped)
		for i := 1; i < len(ids); i++ {
			prefixID := fmt.Sprintf("%016x", fingerprint)
			fingerprint = extendFingerprint(fingerprint, ids[i])
			start, end := spans[i][0], spans[i][1]
			if start < 0 || end > len(prompt) || prompt[start:end] != m.tokenizer.tokenBytes(ids[i]) {
				continue
			}
			record := m.tokenizer.probabilityRecord(m.probRows[i-1], ids[i], start, end)
			record.Position, record.Prefix = i, prefixID
			result.PromptLogprobs = append(result.PromptLogprobs, record)
		}
		if opts.OnLogprobs != nil {
			opts.OnLogprobs(ProbabilityEvent{"prompt", result.PromptLogprobs})
		}
	}
	if prefillOnly {
		result.FinishReason = "prefill"
		return result, nil
	}
	rng := rand.New(rand.NewSource(opts.Seed))
	history := append([]int(nil), ids...)
	decoder := nerdstashDecoder{tokenizer: m.tokenizer}
	output := completionOutput{stops: opts.Stop, onToken: opts.OnToken}
	byteOffset := 0
	for n := 0; n < limit; n++ {
		if err := ctx.Err(); err != nil {
			return GenerateResult{}, err
		}
		samplingLogits := logits
		if opts.Logprobs > 0 && opts.Sampling == nil {
			samplingLogits = append([]float32(nil), logits...)
		}
		token := int(sampleWithOptions(samplingLogits, history, opts, rng))
		history = append(history, token)
		result.CompletionTokens++
		if token == m.config.EOSTokenID {
			result.FinishReason = "stop"
			break
		}
		if opts.Logprobs > 0 {
			piece := m.tokenizer.pieces[token]
			if !m.tokenizer.special[token] && !(len(piece) == 6 && strings.HasPrefix(piece, "<0x")) && !utf8.Valid(decoder.bytes) {
				byteOffset += 2 * len(decoder.bytes)
			}
			row := summarizeLogits(logits, token, opts.Logprobs)
			if len(m.probRows) == len(m.tokens) && len(m.probRows) > 0 {
				m.probRows[len(m.probRows)-1] = row
			}
			end := byteOffset
			if !m.tokenizer.special[token] {
				end += len(m.tokenizer.tokenBytes(token))
			}
			record := m.tokenizer.probabilityRecord(row, token, byteOffset, end)
			record.Position, record.Prefix = len(history)-1, prefixFingerprint(history[:len(history)-1])
			result.Logprobs = append(result.Logprobs, record)
			if opts.OnLogprobs != nil {
				opts.OnLogprobs(ProbabilityEvent{"completion", []TokenLogprob{record}})
			}
			byteOffset = end
		}
		if output.add(decoder.add(token), false) {
			result.FinishReason = "stop"
			break
		}
		if n+1 < limit {
			var err error
			logits, err = m.exchange(ctx, &mlxRequest{len(m.tokens), []int{token}, true})
			if err != nil {
				return GenerateResult{}, err
			}
			m.tokens = append(m.tokens, token)
			if opts.PromptLogprobs {
				m.probRows = append(m.probRows, logprobRow{})
			}
			if opts.OnProgress != nil {
				opts.OnProgress("decode", len(m.tokens))
			}
		}
	}
	if output.add(decoder.flush(), true) {
		result.FinishReason = "stop"
	}
	result.Text = output.text.String()
	return result, nil
}
