package tinyoai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

// NewModelServer serves independent generators under OpenAI model IDs. The
// registry is copied at construction and immutable thereafter. Each generator
// owns its own cache and concurrency policy.
func NewModelServer(models map[string]Generator, options ...ServerOption) (*Server, error) {
	if len(models) == 0 {
		return nil, fmt.Errorf("at least one model is required")
	}
	s := &Server{models: make(map[string]Generator, len(models)), created: time.Now().Unix()}
	for id, model := range models {
		if id == "" || strings.TrimSpace(id) != id || model == nil {
			return nil, fmt.Errorf("invalid model registration %q", id)
		}
		s.models[id] = model
		s.modelIDs = append(s.modelIDs, id)
	}
	for _, option := range options {
		option(s)
	}
	sort.Strings(s.modelIDs)
	return s, nil
}

// SelectModel resolves a request model and writes the API error on failure.
// Optional HTTP handlers use this to share the server's immutable registry.
func (s *Server) SelectModel(w http.ResponseWriter, id *string) (Generator, bool) {
	// Keep the original library constructor compatible with clients that send
	// arbitrary names to a single test model. CLI servers use the strict registry.
	if s.models == nil {
		return s.model, true
	}
	if *id == "" {
		if len(s.modelIDs) != 1 {
			s.writeError(w, http.StatusBadRequest, "model is required; choose a model from /v1/models")
			return nil, false
		}
		*id = s.modelIDs[0]
	}
	if model, ok := s.models[*id]; ok {
		return model, true
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotFound)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
		"message": fmt.Sprintf("model %q is not available; see /v1/models", *id),
		"type":    "invalid_request_error", "param": "model", "code": "model_not_found",
	}})
	return nil, false
}

// listModels writes the immutable registry in sorted model-ID order. A legacy
// single-model NewServer has no advertised registry.
func (s *Server) listModels(w http.ResponseWriter) {
	data := make([]map[string]any, 0, len(s.modelIDs))
	for _, id := range s.modelIDs {
		data = append(data, map[string]any{"id": id, "object": "model", "created": s.created, "owned_by": "local"})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
}
