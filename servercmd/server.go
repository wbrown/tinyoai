// Package servercmd shares model loading and HTTP process lifecycle between commands.
package servercmd

import (
	"context"
	"errors"
	"flag"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/wbrown/tinyoai"
)

// Main parses process flags, loads the selected models, and serves until
// interrupted or its supervising parent exits. A nil factory installs the base
// API; a custom factory composes handlers over the same registry. Startup
// failures terminate the command, and graceful shutdown saves any requested
// CPU cache.
func Main(factory func(map[string]tinyoai.Generator, []string) (http.Handler, error)) {
	addr := flag.String("addr", ":8080", "listen address")
	modelDir := flag.String("model-dir", "", "local Clio directory or GGUF file with tokenizer.json beside it (default: embedded stories260K)")
	mlxPython := flag.String("mlx-python", "", "use this Python with MLX-LM for GPU inference; model-dir must be an MLX model directory")
	mlxNative := flag.Bool("mlx", false, "use native MLX on Apple Silicon (embedded stories260K when model-dir is empty; requires the mlx build tag)")
	cpuModel := flag.String("cpu-model", "", "also load this Go CPU Clio checkpoint as clio-cpu when MLX is enabled")
	threads := flag.Int("threads", 0, "CPU worker limit (0: use GOMAXPROCS / Go runtime default)")
	kvCache := flag.String("kv-cache", "f32", "Clio KV storage precision: f32 or f16 (lossy)")
	cacheFile := flag.String("cache-file", "", "restore Clio prefix state at startup and save it on graceful shutdown")
	readyFile := flag.String("ready-file", "", "write the bound address to this file when ready (process supervisor)")
	parentPID := flag.Int("parent-pid", 0, "stop if this parent process exits (process supervisor)")
	loraPaths := loRAFlags{}
	flag.Var(loraPaths, "lora", "register name=directory for native StableLM LoRA testing (repeatable; enables /v1/adapters)")
	loraActive := flag.String("lora-active", "base", "registered adapter to load initially, or base")
	loraScale := flag.Float64("lora-scale", 1, "initial LoRA strength, multiplied by alpha/rank")
	flag.Parse()
	if len(loraPaths) > 0 && (!*mlxNative || *modelDir == "") {
		log.Fatal("lora requires mlx and a StableLM model-dir")
	}
	if *loraActive != "base" && loraPaths[*loraActive] == "" {
		log.Fatal("lora-active must name a registered adapter or base")
	}
	if *cpuModel != "" && !*mlxNative && *mlxPython == "" {
		log.Fatal("cpu-model requires an MLX backend; for CPU only, use model-dir")
	}
	if *threads < 0 {
		log.Fatal("threads must be nonnegative")
	}
	if *threads > 0 {
		runtime.GOMAXPROCS(*threads)
	}

	var model tinyoai.Generator
	var clio *tinyoai.StableLM
	var err error
	name := "stories260K"
	if *mlxNative && *mlxPython != "" {
		log.Fatal("choose mlx or mlx-python")
	}
	if *mlxNative && *modelDir == "" {
		if *cpuModel != "" || *cacheFile != "" || *kvCache != "f32" {
			log.Fatal("embedded Llama MLX uses float32 KV; CPU checkpoint/cache flags require a Clio model-dir")
		}
		var mlx *tinyoai.LlamaMLX
		mlx, err = tinyoai.DefaultMLX()
		if err == nil {
			defer mlx.Close()
		}
		model = mlx
	} else if *mlxNative || *mlxPython != "" {
		if *modelDir == "" || (*cpuModel == "" && (*cacheFile != "" || *kvCache != "f32")) {
			log.Fatal("MLX requires model-dir; Go cache-file/kv-cache flags require cpu-model")
		}
		name = "clio-accel"
		log.Printf("tinyoai: loading %s from %s", name, *modelDir)
		var mlx *tinyoai.MLX
		if *mlxNative {
			mlx, err = tinyoai.LoadMLXNative(*modelDir)
		} else {
			mlx, err = tinyoai.LoadMLX(*mlxPython, *modelDir)
		}
		if err == nil {
			defer mlx.Close()
		}
		model = mlx
	} else if *modelDir != "" {
		name = "clio-cpu"
		log.Printf("tinyoai: loading %s from %s", name, *modelDir)
		clio, err = tinyoai.LoadStableLMWithOptions(*modelDir, tinyoai.StableLMOptions{KVCache: *kvCache})
		model = clio
	} else {
		if *kvCache != "f32" || *cacheFile != "" {
			log.Fatal("kv-cache and cache-file require model-dir")
		}
		model, err = tinyoai.Default()
	}
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	models := map[string]tinyoai.Generator{name: model}
	modelIDs := []string{name}
	if *cpuModel != "" {
		log.Printf("tinyoai: loading clio-cpu from %s", *cpuModel)
		clio, err = tinyoai.LoadStableLMWithOptions(*cpuModel, tinyoai.StableLMOptions{KVCache: *kvCache})
		if err != nil {
			log.Fatalf("load CPU model: %v", err)
		}
		models["clio-cpu"] = clio
		modelIDs = append(modelIDs, "clio-cpu")
	}
	if *cacheFile != "" {
		if err := clio.LoadCache(*cacheFile); err != nil && !errors.Is(err, os.ErrNotExist) {
			log.Fatalf("restore cache: %v", err)
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if *parentPID > 0 {
		go func() {
			ticker := time.NewTicker(time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					if os.Getppid() != *parentPID {
						stop()
						return
					}
				}
			}
		}()
	}
	var handler http.Handler
	if factory == nil {
		handler, err = tinyoai.NewModelServer(models)
	} else {
		handler, err = factory(models, modelIDs)
	}
	if err != nil {
		log.Fatal(err)
	}
	if len(loraPaths) > 0 {
		adapters := newLoRAHandler(handler, models, loraPaths)
		if *loraActive != "base" {
			info, err := adapters.models[name].LoadLoRA(ctx, loraPaths[*loraActive], *loraScale)
			if err != nil {
				log.Fatalf("load LoRA: %v", err)
			}
			adapters.selected[name] = *loraActive
			log.Printf("tinyoai: LoRA %s sha256=%s scale=%g", *loraActive, info.SHA256, info.Scale)
		}
		handler = adapters
	}
	srv := &http.Server{Addr: *addr, Handler: handler, ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 2 * time.Minute, BaseContext: func(net.Listener) context.Context { return ctx }}
	listener, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	if *readyFile != "" {
		if err := writeReadyFile(*readyFile, listener.Addr().String()); err != nil {
			_ = listener.Close()
			log.Fatalf("server readiness: %v", err)
		}
		defer os.Remove(*readyFile)
	}
	stopped := make(chan struct{})
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdown); err != nil {
			log.Printf("shutdown: %v", err)
			_ = srv.Close() // Release a handler blocked on a slow streaming client.
		}
		close(stopped)
	}()
	log.Printf("tinyoai: serving %s on %s (GET /v1/models; POST /v1/completions or /v1/chat/completions)", strings.Join(modelIDs, ", "), listener.Addr())
	if err := srv.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("serve: %v", err)
	}
	<-stopped
	if *cacheFile != "" {
		if err := clio.SaveCache(*cacheFile); err != nil {
			log.Fatalf("save cache: %v", err)
		}
		log.Printf("tinyoai: saved prefix state to %s", *cacheFile)
	}
}

// writeReadyFile atomically publishes the bound listener address for a
// supervising process. Publication happens after binding, so an unrelated
// service on the requested port cannot be mistaken for this server.
func writeReadyFile(path, address string) error {
	f, err := os.CreateTemp(filepath.Dir(path), ".tinyoai-ready-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	_, err = f.WriteString(address + "\n")
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}
