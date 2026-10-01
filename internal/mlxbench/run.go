//go:build mlx && darwin && arm64 && cgo

package mlxbench

import (
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"time"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

type prefillReference struct {
	Case      string `json:"case"`
	PromptSHA string `json:"prompt_sha256"`
	Tokens    []int  `json:"tokens"`
	Vocab     int    `json:"vocab"`
}

// Run runs an opt-in, resumable diagnostic over
// caller-prepared prompts. Each completed case saves full-vocabulary logits,
// probability rows, timing, and memory telemetry; matching completed cases are
// skipped.
//
// Only one checkpoint is resident at a time. The run changes process-wide
// allocator and optional kernel controls, so it belongs in a dedicated
// diagnostic process rather than concurrent serving. Kernel environment
// overrides are restored on return.
func Run(ctx context.Context, modelDir, out string, config Config, load func(string) (Backend, error)) error {
	if config.MatrixScreen {
		return runMatrixScreen(ctx, modelDir, out, config.Progress, load)
	}
	if config.PreparePrompt == nil {
		return fmt.Errorf("prefill diagnostics require a PreparePrompt context adapter")
	}
	if config.DecodeTokens == 0 {
		config.DecodeTokens = 32
	}
	if config.DecodeTokens < 2 || config.DecodeTokens > 256 || config.CoolSeconds < 0 || config.RestSeconds < 0 {
		return fmt.Errorf("invalid experiment settings")
	}
	names := map[string]bool{}
	for _, c := range config.Cases {
		if !regexp.MustCompile(`^[a-zA-Z0-9_-]+$`).MatchString(c.Name) || names[c.Name] || (c.Context < 2) || c.Chunk < 1 || c.Chunk > 2048 || c.HeadBatch < 1 || c.HeadBatch > 512 || (c.Rollover < 0 || c.Rollover >= c.Context) {
			return fmt.Errorf("invalid experiment case: %+v", c)
		}
		names[c.Name] = true
		if c.QMMFastLoads != "" && c.QMMFastLoads != "stock" && c.QMMFastLoads != "wide" {
			return fmt.Errorf("invalid QMM load mode")
		}
		if c.LayerBatch < 0 || c.LayerBatch > 28 || (c.ProfilePhases && (c.LayerBatch != 0 || c.FusedProjections || c.DensePrefill || c.CaptureTrace)) {
			return fmt.Errorf("invalid phase/layer scheduling case: %+v", c)
		}
		if c.ExactPromptTokens < 0 || (c.ExactPromptTokens > 0 && (c.ExactPromptTokens < 2 || c.ExactPromptTokens+config.DecodeTokens-1 > c.Context || c.Rollover != 0)) {
			return fmt.Errorf("invalid exact prompt length: %+v", c)
		}
		if c.QMMTile != "" {
			parts := strings.Split(c.QMMTile, "x")
			if len(parts) != 3 {
				return fmt.Errorf("invalid QMM tile: %s", c.QMMTile)
			}
			for i, p := range parts {
				if p != "32" && p != "64" && !(i < 2 && p == "128") {
					return fmt.Errorf("invalid QMM tile: %s", c.QMMTile)
				}
			}
			if c.QMMTile == "128x128x64" {
				return fmt.Errorf("QMM tile exceeds the A17 32 KiB threadgroup memory budget")
			}
		}
		if (c.KVBits != 0 && c.KVBits != 8 && c.KVBits != 16) || (c.Model != "" && config.Models[c.Model] == "") || c.CapturePrefix < 0 || (c.CaptureTrace && (c.Rollover != 0 || c.CancelAfterMS != 0 || c.CapturePrefix%c.Chunk != 0)) {
			return fmt.Errorf("invalid precision/capture case: %+v", c)
		}
	}
	if err := os.MkdirAll(out, 0700); err != nil {
		return err
	}
	progress := config.Progress
	if progress == nil {
		progress = func(string) {}
	}
	var m Backend
	var err error
	currentModel := ""
	checkpointIDs := make(map[string]string)
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	executableID, err := fileDigest(ctx, executable)
	if err != nil {
		return err
	}
	var device string
	// Only the opt-in diagnostic sets these process-global experimental knobs.
	// Preserve a launch override for ordinary cases and restore it on return.
	knobs := []string{"TINYOAI_QMM_BM", "TINYOAI_QMM_BN", "TINYOAI_QMM_BK", "TINYOAI_QMM_FAST_LOADS"}
	savedValues, savedPresent := make([]string, len(knobs)), make([]bool, len(knobs))
	for i, key := range knobs {
		savedValues[i], savedPresent[i] = os.LookupEnv(key)
	}
	restoreKnobs := func() {
		for i, key := range knobs {
			if savedPresent[i] {
				_ = os.Setenv(key, savedValues[i])
			} else {
				_ = os.Unsetenv(key)
			}
		}
	}
	defer restoreKnobs()
	defer func() {
		if m != nil {
			m.Close()
		}
	}()
	reset := func() error {
		if err := m.Reset(); err != nil {
			return err
		}
		return mx.Run(func() { mx.ClearCache(); mx.ResetMemory() })
	}
	for _, c := range config.Cases {
		if err := ctx.Err(); err != nil {
			return err
		}
		resultPath := filepath.Join(out, c.Name+".json")
		checkpoint := modelDir
		restoreKnobs()
		if c.QMMFastLoads == "stock" {
			_ = os.Setenv("TINYOAI_QMM_FAST_LOADS", "0")
		} else if c.QMMFastLoads == "wide" {
			_ = os.Setenv("TINYOAI_QMM_FAST_LOADS", "1")
		}
		if c.QMMTile != "" {
			for i, value := range strings.Split(c.QMMTile, "x") {
				if err := os.Setenv(knobs[i], value); err != nil {
					return err
				}
			}
		}
		if c.Model != "" {
			checkpoint = config.Models[c.Model]
		}
		if checkpointIDs[checkpoint] == "" {
			digest, err := checkpointDigest(ctx, checkpoint)
			if err != nil {
				return err
			}
			checkpointIDs[checkpoint] = digest
		}
		if checkpoint != currentModel {
			if m != nil {
				if err := m.Close(); err != nil {
					return err
				}
				m = nil
			}
			progress("load checkpoint " + c.Model)
			m, err = load(checkpoint)
			if err != nil {
				return err
			}
			currentModel = checkpoint
			if err := mx.Run(func() { device = mx.DeviceName() }); err != nil {
				return err
			}
		}
		if err := reset(); err != nil {
			return err
		}
		kvBits := c.KVBits
		if kvBits == 0 {
			kvBits = 8
		}
		if err := m.Configure(c, kvBits); err != nil {
			return err
		}
		cacheMB := c.AllocatorCacheMB
		if cacheMB == 0 {
			cacheMB = 256
			if runtime.GOOS == "ios" {
				cacheMB = 32
			}
		}
		if cacheMB < 1 || cacheMB > 512 {
			return fmt.Errorf("invalid allocator cache limit")
		}
		if err := mx.Run(func() { mx.SetCacheLimit(uint64(cacheMB) << 20) }); err != nil {
			return err
		}
		prompt, err := config.PreparePrompt(m.Tokenizer(), c, config.DecodeTokens)
		if err != nil {
			return err
		}
		if c.ExactPromptTokens > 0 {
			ids := m.Encode(prompt.Prompt)
			if c.ExactPromptTokens > len(ids) {
				return fmt.Errorf("fixture is too short")
			}
			prompt.Prompt, prompt.PromptTokens = m.Decode(ids[:c.ExactPromptTokens]), c.ExactPromptTokens
			if !slices.Equal(m.Encode(prompt.Prompt), ids[:c.ExactPromptTokens]) {
				return fmt.Errorf("exact fixture did not round-trip")
			}
		}
		identity := identityFor(c, config, prompt, checkpointIDs[checkpoint], executableID, device)
		if done, err := completedResult(resultPath, identity); err != nil {
			return err
		} else if done {
			progress("skip completed " + c.Name)
			continue
		}
		opts := PrefillOptions{ContextLength: c.Context, MaxTokens: config.DecodeTokens, KVBits: kvBits, Prefix: prompt.Prefix}
		if c.ExactPromptTokens > 0 {
			// N logit decisions require N-1 further forward passes after prefill.
			opts.MaxTokens = config.DecodeTokens - 1
		}
		if c.Rollover > 0 {
			if prompt.PreviousPrompt == "" || prompt.PreviousPrompt == prompt.Prompt {
				return fmt.Errorf("prefix replacement case requires a different PreviousPrompt")
			}
			progress("seed rollover " + c.Name)
			// Identical old-cache arithmetic for every rollover candidate.
			seed := c
			seed.Chunk, seed.HeadBatch = 512, 32
			if err := m.Configure(seed, kvBits); err != nil {
				return err
			}
			if _, err := m.Prefill(ctx, prompt.PreviousPrompt, opts); err != nil {
				return err
			}
			if err := m.Configure(c, kvBits); err != nil {
				return err
			}
		} else if config.WarmupFull {
			progress("warm full prompt " + c.Name)
			if _, err := m.Prefill(ctx, prompt.Prompt, opts); err != nil {
				return err
			}
			if err := reset(); err != nil {
				return err
			}
		} else {
			// Warm this projection/selection shape, then discard KV and allocator
			// cache. Weight loading and warm-up are outside the timed request.
			ids := m.Encode(prompt.Prompt)
			warm := min(c.Chunk, len(ids)-1)
			targets := append([]int(nil), ids[1:warm+1]...)
			if err := m.Warmup(ctx, ids[:warm], targets); err != nil {
				return err
			}
			if err := reset(); err != nil {
				return err
			}
		}
		if config.RestSeconds > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Duration(config.RestSeconds) * time.Second):
			}
		}
		for deadline := time.Now().Add(time.Duration(config.CoolSeconds) * time.Second); mx.ReadProcessMetrics().ThermalState > 0 && time.Now().Before(deadline); {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Second):
			}
		}
		if err := mx.Run(mx.ResetMemory); err != nil {
			return err
		}
		progress("start " + c.Name)
		if c.ReserveKV {
			m.Reserve(((prompt.PromptTokens + 255) / 256) * 256)
		}
		if c.CaptureTrace {
			if c.CapturePrefix >= prompt.PromptTokens-1 {
				return fmt.Errorf("capture prefix outside prompt")
			}
			m.Capture(filepath.Join(out, c.Name+".gputrace"), c.CapturePrefix)
		}
		report, err := runPrefillCase(ctx, m, prompt.Prompt, opts, c, out, config)
		if err != nil {
			_ = WriteJSON(filepath.Join(out, c.Name+"-error.json"), map[string]any{"case": c, "error": err.Error(), "report": report})
			return fmt.Errorf("%s: %w", c.Name, err)
		}
		report["identity"] = identity
		if err := WriteJSON(resultPath, report); err != nil {
			return err
		}
		progress(fmt.Sprintf("done %s prefill=%.3fs transformer=%.3fs head=%.3fs peak=%.3fGB KL=%.9f argmax=%d/%d", c.Name, report["prefill_seconds"], report["transformer_seconds"], report["probability_seconds"], float64(report["peak_process_bytes"].(uint64))/1e9, report["mean_kl"], report["argmax_agreement"], config.DecodeTokens))
	}
	return nil
}

// runPrefillCase measures prefill and forced-token decode against a saved
// reference for the same prompt and context scenario. It optionally interrupts
// and resumes prefill, samples process telemetry, and writes full logits and
// probability rows. The first approved baseline establishes the reference;
// later cases compare argmax and full-vocabulary KL on identical token
// histories.
func runPrefillCase(ctx context.Context, m Backend, prompt string, opts PrefillOptions, c Case, out string, config Config) (map[string]any, error) {
	ids := m.Encode(prompt)
	promptSHA := fmt.Sprintf("%x", sha256.Sum256([]byte(prompt)))
	scenario := fmt.Sprintf("c%d-r%d", c.Context, c.Rollover)
	if c.ExactPromptTokens > 0 {
		scenario += fmt.Sprintf("-p%d", c.ExactPromptTokens)
	}
	refPath := filepath.Join(out, "reference-"+scenario+".json")
	var ref prefillReference
	var reference []float32
	if data, err := os.ReadFile(refPath); err == nil {
		if err := json.Unmarshal(data, &ref); err != nil {
			return nil, err
		}
		if ref.PromptSHA != promptSHA || len(ref.Tokens) != config.DecodeTokens || ref.Vocab != m.Tokenizer().VocabSize {
			return nil, fmt.Errorf("reference mismatch")
		}
		f, err := os.Open(filepath.Join(out, ref.Case+".f32"))
		if err != nil {
			return nil, err
		}
		reference = make([]float32, config.DecodeTokens*ref.Vocab)
		err = binary.Read(f, binary.LittleEndian, reference)
		_ = f.Close()
		if err != nil {
			return nil, err
		}
	} else if !os.IsNotExist(err) {
		return nil, err
	} else if config.ReferenceCase != "" {
		if c.Name != config.ReferenceCase {
			return nil, fmt.Errorf("run reference case %s first", config.ReferenceCase)
		}
	} else if c.Chunk != 512 || c.HeadBatch != 32 || c.TopKBlock != 0 || c.ReserveKV || c.FusedProjections || c.DensePrefill || c.AllocatorCacheMB != 0 || c.LayerBatch != 0 || c.ProfilePhases || c.OutputRoot || c.QMMTile != "" || c.QMMFastLoads != "" {
		return nil, fmt.Errorf("run a 512/32 baseline first for %s", scenario)
	}
	chunkFile, err := os.Create(filepath.Join(out, c.Name+"-chunks.jsonl"))
	if err != nil {
		return nil, err
	}
	defer chunkFile.Close()
	encoder := json.NewEncoder(chunkFile)
	var writeErr error
	phaseTotals := map[string]float64{}
	var observePhase func(Phase)
	if c.ProfilePhases {
		phaseFile, e := os.Create(filepath.Join(out, c.Name+"-phases.jsonl"))
		if e != nil {
			return nil, e
		}
		defer phaseFile.Close()
		phaseEncoder := json.NewEncoder(phaseFile)
		observePhase = func(p Phase) {
			phaseTotals[p.Name] += p.Seconds
			if e := phaseEncoder.Encode(p); e != nil {
				writeErr = e
			}
		}
	}
	var chunks []Chunk
	var logits []float32
	m.Observe(func(p Chunk) {
		logits = p.Logits
		p.Logits = nil
		chunks = append(chunks, p)
		if e := encoder.Encode(p); e != nil {
			writeErr = e
		}
	}, observePhase)
	defer m.Observe(nil, nil)
	startMetrics := mx.ReadProcessMetrics()
	warnings := 0
	if config.MemoryWarnings != nil {
		warnings = config.MemoryWarnings()
	}
	var samples []mx.ProcessMetrics
	stop, stopped := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(stopped)
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		for {
			samples = append(samples, mx.ReadProcessMetrics())
			select {
			case <-stop:
				return
			case <-ticker.C:
			}
		}
	}()
	started := time.Now()
	var cachedTokens int
	cancelLatency, completedAtCancel, reusedAfterCancel := 0.0, 0, 0
	if c.CancelAfterMS > 0 {
		cancelCtx, cancel := context.WithCancel(ctx)
		requested := make(chan time.Time, 1)
		var timer *time.Timer
		opts.OnProgress = func(_ string, _ int) {
			if timer == nil {
				timer = time.AfterFunc(time.Duration(c.CancelAfterMS)*time.Millisecond, func() {
					requested <- time.Now()
					cancel()
				})
			}
		}
		_, err = m.Prefill(cancelCtx, prompt, opts)
		if timer != nil {
			timer.Stop()
		}
		cancel()
		if errors.Is(err, context.Canceled) && ctx.Err() == nil {
			cancelLatency = time.Since(<-requested).Seconds()
			var rowCount int
			completedAtCancel, rowCount = m.CacheSize()
			if rowCount != completedAtCancel {
				err = fmt.Errorf("cancellation lost probability rows")
			} else {
				opts.OnProgress = nil
				cachedTokens, err = m.Prefill(ctx, prompt, opts)
				reusedAfterCancel = cachedTokens
				if err == nil && reusedAfterCancel < completedAtCancel-1 {
					err = fmt.Errorf("cancellation lost cached prefix")
				}
			}
		} else if err == nil {
			err = fmt.Errorf("cancellation did not interrupt the request")
		}
	} else {
		cachedTokens, err = m.Prefill(ctx, prompt, opts)
	}
	prefillSeconds := time.Since(started).Seconds()
	close(stop)
	<-stopped
	if err != nil {
		return nil, err
	}
	if writeErr != nil {
		return nil, writeErr
	}
	_, rowCount := m.CacheSize()
	if len(logits) != m.Tokenizer().VocabSize || rowCount != len(ids) {
		return nil, fmt.Errorf("incomplete logits/probability rows")
	}
	var active, peak uint64
	if err := mx.Run(func() { active, peak = mx.Memory() }); err != nil {
		return nil, err
	}
	var peakProcess uint64
	maxThermal := startMetrics.ThermalState
	for _, s := range samples {
		peakProcess = max(peakProcess, s.FootprintBytes)
		maxThermal = max(maxThermal, s.ThermalState)
	}
	transformer, probability, longest := 0.0, 0.0, 0.0
	for _, p := range chunks {
		transformer += p.TransformerSeconds
		probability += p.ProbabilitySeconds
		longest = max(longest, p.TotalSeconds)
	}
	var allLogits []float32
	var forced []int
	var stepKL []float64
	agree, meanKL, maxKL := 0, 0.0, 0.0
	decodeSeconds := 0.0
	m.Observe(nil, nil)
	for i := 0; i < config.DecodeTokens; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		best := Argmax(logits)
		chosen := best
		kl := 0.0
		if reference != nil {
			chosen = ref.Tokens[i]
			kl = KL(reference[i*ref.Vocab:(i+1)*ref.Vocab], logits)
		}
		if best == chosen {
			agree++
		}
		meanKL += kl
		maxKL = max(maxKL, kl)
		stepKL = append(stepKL, kl)
		forced = append(forced, chosen)
		allLogits = append(allLogits, logits...)
		if i+1 < config.DecodeTokens {
			t := time.Now()
			logits, err = m.Forward(ctx, len(ids)+i, []int{chosen})
			decodeSeconds += time.Since(t).Seconds()
			if err != nil {
				return nil, err
			}
		}
	}
	logitFile, err := os.Create(filepath.Join(out, c.Name+".f32"))
	if err != nil {
		return nil, err
	}
	err = binary.Write(logitFile, binary.LittleEndian, allLogits)
	closeErr := logitFile.Close()
	if err != nil {
		return nil, err
	}
	if closeErr != nil {
		return nil, closeErr
	}
	rowsFile, err := os.Create(filepath.Join(out, c.Name+"-probabilities.json.gz"))
	if err != nil {
		return nil, err
	}
	z := gzip.NewWriter(rowsFile)
	err = m.WriteProbabilities(z)
	zerr := z.Close()
	cerr := rowsFile.Close()
	if err != nil {
		return nil, err
	}
	if zerr != nil {
		return nil, zerr
	}
	if cerr != nil {
		return nil, cerr
	}
	if reference == nil {
		ref = prefillReference{c.Name, promptSHA, forced, m.Tokenizer().VocabSize}
		if err := WriteJSON(refPath, ref); err != nil {
			return nil, err
		}
	}
	if config.MemoryWarnings != nil {
		warnings = config.MemoryWarnings() - warnings
	}
	report := map[string]any{"complete": true, "case": c, "started_utc": started.UTC(), "platform": runtime.GOOS, "decode_tokens": config.DecodeTokens, "prompt_sha256": promptSHA, "prompt_ids": ids, "forced_tokens": forced, "reference_case": ref.Case, "prefill_seconds": prefillSeconds, "transformer_seconds": transformer, "probability_seconds": probability, "longest_chunk_seconds": longest, "prefill_tps": float64(len(ids)-cachedTokens) / prefillSeconds, "prompt_tokens": len(ids), "cached_tokens": cachedTokens, "decode_tps": float64(config.DecodeTokens-1) / decodeSeconds, "active_mlx_bytes": active, "peak_mlx_bytes": peak, "peak_process_bytes": peakProcess, "memory_warnings": warnings, "start_metrics": startMetrics, "max_thermal_state": maxThermal, "samples": samples, "chunks": chunks, "argmax_agreement": agree, "mean_kl": meanKL / float64(config.DecodeTokens), "max_kl": maxKL, "step_kl": stepKL, "weight_bits": m.WeightBits(), "kv_bits": opts.KVBits, "probability_rows": rowCount}
	if c.ProfilePhases {
		report["phase_seconds"] = phaseTotals
	}
	report["last_kv_position"] = len(ids) + config.DecodeTokens - 2
	if err := mx.Run(func() { report["metal_device"] = mx.DeviceName() }); err != nil {
		return nil, err
	}
	if len(chunks) > 0 {
		report["cached_tokens"] = chunks[0].Prefix
		report["prefill_tps"] = float64(len(ids)-chunks[0].Prefix) / prefillSeconds
	}
	if c.CancelAfterMS > 0 {
		report["cancellation_latency_seconds"], report["completed_at_cancel"], report["reused_after_cancel"] = cancelLatency, completedAtCancel, reusedAfterCancel
	}
	return report, nil
}
