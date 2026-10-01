package tinyoai

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// Server adapts generators to model discovery and OpenAI-compatible text/chat
// completions with JSON or SSE responses. It handles requests concurrently;
// generators must provide their own state isolation or serialization. Configure
// options before serving and close owned generators after requests have stopped.
type Server struct {
	model      Generator
	models     map[string]Generator
	modelIDs   []string
	created    int64
	counter    atomic.Int64
	extensions []CompletionExtension
}

// Generator is an inference backend. Implementations used by Server must allow
// concurrent calls, either using separate state or serializing inference.
type Generator interface {
	// Generate completes a prompt, honoring cancellation and emitting callbacks
	// synchronously before returning.
	Generate(string, GenerateOptions) (GenerateResult, error)
}

// Prefiller optionally prepares reusable context without sampling a completion.
type Prefiller interface {
	// Prefill prepares prompt state and optional probabilities without sampling
	// or emitting completion text.
	Prefill(string, GenerateOptions) (GenerateResult, error)
}

// NewServer wraps one concurrently usable generator and applies options in
// order. It accepts any request model name for compatibility; use
// NewModelServer for validated model IDs and discovery. The caller retains
// ownership of the generator and its lifecycle.
func NewServer(m Generator, options ...ServerOption) *Server {
	s := &Server{model: m}
	for _, option := range options {
		option(s)
	}
	return s
}

// NewDefaultServer returns a Server backed by the embedded stories260K model.
func NewDefaultServer(options ...ServerOption) (*Server, error) {
	m, err := Default()
	if err != nil {
		return nil, err
	}
	return NewServer(m, options...), nil
}

// ChatMessage is one message in a chat-completions request. Content is left raw
// because the OpenAI spec allows either a string or an array of content parts.
type ChatMessage struct {
	// Role is accepted for wire compatibility but does not affect prompt
	// construction.
	Role string `json:"role"`
	// Content contains a string or an array of content parts; only textual
	// content is retained.
	Content json.RawMessage `json:"content"`
	// ToolCallID is accepted but not interpreted.
	ToolCallID string `json:"tool_call_id,omitempty"`
	// ToolCalls is accepted but does not trigger tool execution.
	ToolCalls json.RawMessage `json:"tool_calls,omitempty"`
}

// ChatRequest is the subset of the chat-completions request body this server
// understands. Unknown fields are ignored.
type ChatRequest struct {
	// Model selects the generator when a named registry is used.
	Model string `json:"model"`
	// Messages are flattened into a plain text prompt without a role template.
	Messages []ChatMessage `json:"messages"`
	// MaxCompletionTokens sets the output limit and takes precedence over
	// MaxTokens when nonzero.
	MaxCompletionTokens int `json:"max_completion_tokens"`
	// MaxTokens is the legacy output-limit field; the chat default is 256 when
	// both limits are zero.
	MaxTokens int `json:"max_tokens"`
	// Temperature sets sampling temperature; nil selects one and zero selects
	// greedy decoding.
	Temperature *float64 `json:"temperature"`
	// TopP is accepted for compatibility but is not applied by the chat adapter.
	TopP *float64 `json:"top_p"`
	// Seed fixes sampling randomness; nil requests a fresh seed.
	Seed *int64 `json:"seed"`
	// Stop lists stop strings using this adapter's array-only representation.
	Stop []string `json:"stop"`
	// Stream selects SSE response chunks.
	Stream bool `json:"stream"`
	// StreamOptions optionally requests a final usage chunk.
	StreamOptions *streamOptions `json:"stream_options"`
	// Tools is accepted but does not enable tool calling.
	Tools json.RawMessage `json:"tools,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// ServeHTTP routes model listing, text completions, and chat completions. Chat
// message text is flattened before generation; optional completion extensions
// apply only to the text endpoint.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
		s.listModels(w)
		return
	}
	if r.Method == http.MethodPost && r.URL.Path == "/v1/completions" {
		s.completions(w, r)
		return
	}
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		http.NotFound(w, r)
		return
	}

	raw, err := readCompletionBody(w, r)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("read body: %v", err))
		return
	}

	var req ChatRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("decode request: %v", err))
		return
	}
	model, ok := s.SelectModel(w, &req.Model)
	if !ok {
		return
	}

	prompt := flattenMessages(req.Messages)
	opts := s.options(req)
	opts.Context = r.Context()
	if err := validateGeneration(opts); err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	if req.Stream {
		s.streamResponse(w, req, prompt, opts, model)
		return
	}
	s.jsonResponse(w, req, prompt, opts, model)
}

// options maps legacy chat controls to generation settings, defaulting to 256
// output tokens, temperature 1, and a fresh seed. Chat top_p and tool fields
// are accepted but not applied.
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
func (s *Server) jsonResponse(w http.ResponseWriter, req ChatRequest, prompt string, opts GenerateOptions, model Generator) {
	result, err := model.Generate(prompt, opts)
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
func (s *Server) streamResponse(w http.ResponseWriter, req ChatRequest, prompt string, opts GenerateOptions, model Generator) {
	ctx, cancel := context.WithCancel(opts.Context)
	defer cancel()
	opts.Context = ctx
	stream, err := newEventStream(w, cancel)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

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
		stream.send(chunk)
	}

	// Opening role chunk, then a content chunk per token.
	sendChunk(map[string]any{"role": "assistant"}, nil)
	if stream.err != nil {
		return
	}
	opts.OnToken = func(piece string) {
		sendChunk(map[string]any{"content": piece}, nil)
	}

	result, err := model.Generate(prompt, opts)
	if err != nil {
		// The stream has already started; surface the error as a final event.
		stream.send(map[string]any{"error": map[string]any{"message": err.Error()}})
		stream.write("[DONE]")
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
		stream.send(chunk)
	}
	stream.write("[DONE]")
}

// id returns a completion identifier unique within this Server, using an
// atomic counter for concurrent requests.
func (s *Server) id() string {
	return fmt.Sprintf("chatcmpl-tiny-%d", s.counter.Add(1))
}

// writeError sends a JSON invalid-request error with the supplied HTTP status.
// Call it before a streaming response has committed its headers.
func (s *Server) writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": "invalid_request_error"},
	})
}

// usageJSON returns the engine result's standard token-accounting object for
// either completion endpoint.
func usageJSON(r GenerateResult) map[string]any { return r.Usage() }

// Usage renders token accounting in the OpenAI usage shape.
func (r GenerateResult) Usage() map[string]any {
	usage := map[string]any{
		"prompt_tokens":     r.PromptTokens,
		"completion_tokens": r.CompletionTokens,
		"total_tokens":      r.PromptTokens + r.CompletionTokens,
	}
	if r.CachedPromptTokens > 0 {
		usage["prompt_tokens_details"] = map[string]any{"cached_tokens": r.CachedPromptTokens}
	}
	return usage
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
