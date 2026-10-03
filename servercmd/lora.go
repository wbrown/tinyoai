package servercmd

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"sort"
	"strings"

	"github.com/wbrown/tinyoai"
)

// loRAFlags registers local directories before HTTP starts. Clients select a
// registered name, never a path; base is reserved for unloading the adapter.
type loRAFlags map[string]string

// String formats names for flag diagnostics without exposing local paths.
func (f loRAFlags) String() string {
	var names []string
	for name := range f {
		names = append(names, name)
	}
	sort.Strings(names)
	return strings.Join(names, ",")
}

// Set accepts repeated name=directory flags and rejects ambiguous selections.
func (f loRAFlags) Set(value string) error {
	name, dir, ok := strings.Cut(value, "=")
	if !ok || name == "" || name == "base" || len(name) > 128 || strings.ContainsAny(name, " /\\\t\r\n") || dir == "" || f[name] != "" {
		return fmt.Errorf("use a unique LoRA name=directory; base is reserved")
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	f[name] = abs
	return nil
}

type loRASelection struct {
	Adapter string           `json:"adapter"`
	Info    tinyoai.LoRAInfo `json:"info"`
}

// loRAHandler is an opt-in experiment endpoint around the ordinary API.
// Selection is session-wide, not a per-request OpenAI parameter. Its gate keeps
// selection names and model updates together; the model's gate waits for active
// inference. Both waits must allow cancellation because inference can be long.
type loRAHandler struct {
	next     http.Handler
	models   map[string]tinyoai.LoRAController
	paths    loRAFlags
	selected map[string]string
	gate     chan struct{}
}

// newLoRAHandler exposes adapter controls only for supplied capable models.
func newLoRAHandler(next http.Handler, models map[string]tinyoai.Generator, paths loRAFlags) *loRAHandler {
	h := &loRAHandler{
		next: next, paths: paths,
		models:   make(map[string]tinyoai.LoRAController),
		selected: make(map[string]string),
		gate:     make(chan struct{}, 1),
	}
	h.gate <- struct{}{}
	for id, model := range models {
		if controller, ok := model.(tinyoai.LoRAController); ok {
			h.models[id] = controller
			h.selected[id] = "base"
		}
	}
	return h
}

// ServeHTTP lists adapter identity or changes one idle/queued model selection.
// Malformed requests are rejected before touching the model. No prompt template
// is inferred from an adapter; clients continue to supply complete prompts.
func (h *loRAHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/v1/adapters" {
		h.next.ServeHTTP(w, r)
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		w.Header().Set("Allow", "GET, POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	select {
	case <-r.Context().Done():
		http.Error(w, r.Context().Err().Error(), http.StatusBadRequest)
		return
	case <-h.gate:
	}
	defer func() { h.gate <- struct{}{} }()
	// A ready gate and cancellation can both win the select. Check again before
	// reading request data or calling a model that might commit a selection.
	if err := r.Context().Err(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if r.Method == http.MethodGet {
		var names []string
		for name := range h.paths {
			names = append(names, name)
		}
		sort.Strings(names)
		states := make(map[string]loRASelection)
		for id, model := range h.models {
			info, err := model.LoRAInfo(r.Context())
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			states[id] = loRASelection{Adapter: h.selected[id], Info: info}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"adapters": append([]string{"base"}, names...), "models": states})
		return
	}
	var request struct {
		Model   string   `json:"model"`
		Adapter string   `json:"adapter"`
		Scale   *float64 `json:"scale"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096))
	decoder.DisallowUnknownFields()
	err := decoder.Decode(&request)
	if err == nil {
		var extra any
		if decoder.Decode(&extra) != io.EOF {
			err = fmt.Errorf("expected one JSON object")
		}
	}
	if err != nil || h.models[request.Model] == nil || (request.Adapter != "base" && h.paths[request.Adapter] == "") {
		http.Error(w, "expected model, registered adapter, and optional numeric scale", http.StatusBadRequest)
		return
	}
	scale := 1.0
	if request.Scale != nil {
		scale = *request.Scale
	}
	model := h.models[request.Model]
	var info tinyoai.LoRAInfo
	if request.Adapter != "base" && h.selected[request.Model] == request.Adapter {
		info, err = model.SetLoRAScale(r.Context(), scale)
	} else {
		info, err = model.LoadLoRA(r.Context(), h.paths[request.Adapter], scale)
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	h.selected[request.Model] = request.Adapter
	_ = json.NewEncoder(w).Encode(loRASelection{Adapter: request.Adapter, Info: info})
}
