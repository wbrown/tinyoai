package tinyoai

import (
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// Server is an OpenAI-compatible /v1/chat/completions endpoint backed by a
// tinyoai Model. It is a real inference server: requests are decoded, flattened
// to a prompt, and run through the transformer. Responses (JSON and SSE) carry
// real token counts and finish reasons. Generate is safe for concurrent use, so
// a single Server can handle requests concurrently.
type Server struct {
	model   *Model
	counter atomic.Int64
}

// NewServer returns a Server backed by the given model.
func NewServer(m *Model) *Server {
	return &Server{model: m}
}

// NewDefaultServer returns a Server backed by the embedded stories260K model.
func NewDefaultServer() (*Server, error) {
	m, err := Default()
	if err != nil {
		return nil, err
	}
	return NewServer(m), nil
}

// ChatMessage is one message in a chat-completions request. Content is left raw
// because the OpenAI spec allows either a string or an array of content parts.
type ChatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
}

// ChatRequest is the subset of the chat-completions request body this server
// understands. Unknown fields are ignored.
type ChatRequest struct {
	Model               string          `json:"model"`
	Messages            []ChatMessage   `json:"messages"`
	MaxCompletionTokens int             `json:"max_completion_tokens"`
	MaxTokens           int             `json:"max_tokens"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	Seed                *int64          `json:"seed"`
	Stop                []string        `json:"stop"`
	Stream              bool            `json:"stream"`
	StreamOptions       *streamOptions  `json:"stream_options"`
	Tools               json.RawMessage `json:"tools,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// ServeHTTP implements http.Handler, routing POST /v1/chat/completions.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		http.NotFound(w, r)
		return
	}

	raw, err := io.ReadAll(r.Body)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("read body: %v", err))
		return
	}

	var req ChatRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("decode request: %v", err))
		return
	}

	prompt := flattenMessages(req.Messages)
	opts := s.options(req)

	if req.Stream {
		s.streamResponse(w, req, prompt, opts)
		return
	}
	s.jsonResponse(w, req, prompt, opts)
}

// options builds GenerateOptions from a request, applying server defaults.
func (s *Server) options(req ChatRequest) GenerateOptions {
	maxTokens := req.MaxCompletionTokens
	if maxTokens == 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens == 0 {
		maxTokens = 256
	}
	temperature := 1.0
	if req.Temperature != nil {
		temperature = *req.Temperature
	}
	var seed int64
	if req.Seed != nil {
		seed = *req.Seed
	} else {
		seed = rand.Int63()
	}
	return GenerateOptions{
		MaxTokens:   maxTokens,
		Temperature: temperature,
		Seed:        seed,
		Stop:        req.Stop,
	}
}

// jsonResponse generates the full completion and writes a single JSON response.
func (s *Server) jsonResponse(w http.ResponseWriter, req ChatRequest, prompt string, opts GenerateOptions) {
	result, err := s.model.Generate(prompt, opts)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	resp := map[string]any{
		"id":      s.id(),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   req.Model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": result.Text},
			"finish_reason": result.FinishReason,
		}},
		"usage": usageJSON(result),
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// streamResponse generates token-by-token and writes an SSE stream that mirrors
// OpenAI's chat.completion.chunk events.
func (s *Server) streamResponse(w http.ResponseWriter, req ChatRequest, prompt string, opts GenerateOptions) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.writeError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := s.id()
	created := time.Now().Unix()
	sendChunk := func(delta map[string]any, finish any) {
		chunk := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   req.Model,
			"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}},
		}
		writeSSE(w, chunk)
		flusher.Flush()
	}

	// Opening role chunk, then a content chunk per token.
	sendChunk(map[string]any{"role": "assistant"}, nil)
	opts.OnToken = func(piece string) {
		sendChunk(map[string]any{"content": piece}, nil)
	}

	result, err := s.model.Generate(prompt, opts)
	if err != nil {
		// The stream has already started; surface the error as a final event.
		writeSSE(w, map[string]any{"error": map[string]any{"message": err.Error()}})
		flusher.Flush()
		writeRaw(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}

	sendChunk(map[string]any{}, result.FinishReason)
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		chunk := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   req.Model,
			"choices": []map[string]any{},
			"usage":   usageJSON(result),
		}
		writeSSE(w, chunk)
		flusher.Flush()
	}
	writeRaw(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *Server) id() string {
	return fmt.Sprintf("chatcmpl-tiny-%d", s.counter.Add(1))
}

func (s *Server) writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": "invalid_request_error"},
	})
}

// usageJSON renders token accounting in OpenAI's usage shape.
func usageJSON(r GenerateResult) map[string]any {
	return map[string]any{
		"prompt_tokens":     r.PromptTokens,
		"completion_tokens": r.CompletionTokens,
		"total_tokens":      r.PromptTokens + r.CompletionTokens,
	}
}

func writeSSE(w http.ResponseWriter, v any) {
	data, err := json.Marshal(v)
	if err != nil {
		return
	}
	writeRaw(w, "data: "+string(data)+"\n\n")
}

func writeRaw(w http.ResponseWriter, s string) {
	_, _ = w.Write([]byte(s))
}

// flattenMessages concatenates message text content into a single prompt. The
// stories260K model is a plain text model with no chat or role structure, so
// roles are dropped and the textual content is joined with spaces.
func flattenMessages(messages []ChatMessage) string {
	var parts []string
	for _, m := range messages {
		if text := contentText(m.Content); text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, " ")
}

// contentText extracts plain text from a message Content value, which may be a
// JSON string or an array of typed content parts.
func contentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var b strings.Builder
		for _, p := range parts {
			if p.Type == "text" {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return ""
}
