package tinyoai

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestCacheFile round-trips completed FP32 and FP16 prefix chunks, checks
// restored inference, and rejects damaged or incompatible snapshots without
// replacing a valid resident cache.
func TestCacheFile(t *testing.T) {
	for _, dtype := range []string{"f32", "f16"} {
		t.Run(dtype, func(t *testing.T) {
			load := func() *StableLM {
				t.Helper()
				m, err := LoadStableLMWithOptions("testdata/stablelm", StableLMOptions{KVCache: dtype})
				if err != nil {
					t.Fatal(err)
				}
				m.config.MaxPositionEmbeddings = 384
				return m
			}
			m := load()
			path := filepath.Join(t.TempDir(), "prefix.kv")
			if err := m.SaveCache(path); err != nil {
				t.Fatal(err)
			}
			if err := m.LoadCache(path); err != nil {
				t.Fatal(err)
			}
			prompt := "Hello" + strings.Repeat("1", prefillBatch+26)
			ctx := &cancelAfterChecks{Context: context.Background(), remaining: 6}
			if _, err := m.Generate(prompt, GenerateOptions{MaxTokens: 4, Context: ctx}); err != context.Canceled {
				t.Fatalf("cancel: %v", err)
			}
			saved := <-m.cache
			if len(saved.tokens) != prefillBatch {
				t.Fatalf("completed prefix length %d", len(saved.tokens))
			}
			m.cache <- saved
			if err := m.SaveCache(path); err != nil {
				t.Fatal(err)
			}
			stat, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			if stat.Mode().Perm() != 0600 {
				t.Fatalf("cache permissions %v", stat.Mode())
			}
			restored := load()
			if err := restored.LoadCache(path); err != nil {
				t.Fatal(err)
			}
			got, err := restored.Generate(prompt, GenerateOptions{MaxTokens: 8})
			if err != nil {
				t.Fatal(err)
			}
			if got.CachedPromptTokens != prefillBatch {
				t.Fatalf("restored reuse %d", got.CachedPromptTokens)
			}
			fresh := load()
			want, err := fresh.Generate(prompt, GenerateOptions{MaxTokens: 8})
			if err != nil {
				t.Fatal(err)
			}
			want.CachedPromptTokens = got.CachedPromptTokens
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("restored %+v vs cold %+v", got, want)
			}
			before := <-restored.cache
			cold := <-fresh.cache
			if kl := distributionKL(cold.state.logits, before.state.logits); math.IsNaN(kl) || kl > 1e-8 {
				t.Fatalf("restored KL %g", kl)
			}
			restored.cache <- before
			original, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			corrupt := append([]byte(nil), original...)
			header := int(binary.LittleEndian.Uint32(corrupt[8:12]))
			corrupt[12+header+10] ^= 1
			for _, data := range [][]byte{original[:10], original[:len(original)-1], append(append([]byte(nil), original...), 0), corrupt} {
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
				if err := restored.LoadCache(path); err == nil {
					t.Fatal("accepted damaged cache")
				}
				after := <-restored.cache
				if after != before {
					t.Fatal("failed restore replaced valid cache")
				}
				restored.cache <- after
			}
			if err := os.WriteFile(path, original, 0600); err != nil {
				t.Fatal(err)
			}
			other := load()
			other.layers[0].q.data[0] ^= 1
			if err := other.LoadCache(path); err == nil {
				t.Fatal("accepted different checkpoint")
			}
			other = load()
			other.halfKV = !other.halfKV
			if err := other.LoadCache(path); err == nil {
				t.Fatal("accepted different KV precision")
			}
		})
	}
}

// TestClioCacheFile is an opt-in full-model persistence measurement. It
// discards the resident cache and identity before restore so restore timing
// includes checkpoint hashing, then compares output and full-vocabulary KL.
func TestClioCacheFile(t *testing.T) {
	model, ref := os.Getenv("TINYOAI_CLIO_DIR"), os.Getenv("TINYOAI_CACHE_REFERENCE")
	if model == "" || ref == "" {
		t.Skip("set TINYOAI_CLIO_DIR and TINYOAI_CACHE_REFERENCE")
	}
	m, err := LoadStableLMWithOptions(model, StableLMOptions{KVCache: os.Getenv("TINYOAI_KV_CACHE")})
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(ref, "prompt.txt"))
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	cold, err := m.Generate(string(data), GenerateOptions{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	coldSeconds := time.Since(start).Seconds()
	saved := <-m.cache
	want := append([]float32(nil), saved.state.logits...)
	m.cache <- saved
	path := os.Getenv("TINYOAI_CACHE_FILE")
	if path == "" {
		path = filepath.Join(t.TempDir(), "clio.kv")
	}
	start = time.Now()
	if err := m.SaveCache(path); err != nil {
		t.Fatal(err)
	}
	saveSeconds := time.Since(start).Seconds()
	<-m.cache
	m.cache <- nil
	saved = nil
	m.cacheIdentity = [32]byte{}
	runtime.GC()
	start = time.Now()
	if err := m.LoadCache(path); err != nil {
		t.Fatal(err)
	}
	loadSeconds := time.Since(start).Seconds()
	start = time.Now()
	got, err := m.Generate(string(data), GenerateOptions{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	warmSeconds := time.Since(start).Seconds()
	cold.CachedPromptTokens = got.CachedPromptTokens
	if !reflect.DeepEqual(got, cold) || got.CachedPromptTokens != got.PromptTokens-1 {
		t.Fatalf("restored %+v vs cold %+v", got, cold)
	}
	saved = <-m.cache
	actual := append([]float32(nil), saved.state.logits...)
	m.cache <- saved
	kl := distributionKL(want, actual)
	if math.IsNaN(kl) || kl > 1e-7 {
		t.Fatalf("restored KL %g", kl)
	}
	trace := newLogitTrace(t, os.Getenv("TINYOAI_CACHE_TRACE"), m.config.VocabSize)
	trace.append(t, want)
	trace.append(t, actual)
	stat, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	report := map[string]any{"model": model, "kv_cache_float16": m.halfKV, "cold_seconds": coldSeconds, "save_seconds": saveSeconds, "restore_seconds": loadSeconds, "warm_seconds": warmSeconds, "cached_tokens": got.CachedPromptTokens, "snapshot_bytes": stat.Size(), "kl": kl, "memory": parityMemory()}
	t.Logf("persistent prefix: %+v", report)
	if path := os.Getenv("TINYOAI_CACHE_REPORT"); path != "" {
		data, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(path, append(data, '\n'), 0600); err != nil {
			t.Fatal(err)
		}
	}
}
