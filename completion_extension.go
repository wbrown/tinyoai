package tinyoai

import "encoding/json"

// CompletionExtension adapts additional completion fields and events. Hooks are
// installed explicitly and must be safe for concurrent requests. Decode runs
// before generation; Result and Event may add application-specific wire data.
// Result decorates JSON responses and the final streaming usage (only usage
// changes are retained in a usage chunk). Event encodes extra SSE frames.
// Inference, cancellation, ordinary responses and SSE remain owned by Server.
type CompletionExtension struct {
	// Decode reads additional request fields and may alter options or reject the
	// request before inference.
	Decode func(json.RawMessage, *GenerateOptions) error
	// Result decorates the ordinary response or requested streaming usage. Only
	// usage changes survive in a usage chunk.
	Result func(GenerateOptions, GenerateResult, map[string]any)
	// Event returns an additional JSON-encodable SSE payload, or nil to omit it.
	// It does not replace standard probability output.
	Event func(ProbabilityEvent) any
}

// ServerOption configures a server at construction time, before concurrent
// requests begin.
type ServerOption func(*Server)

// WithCompletionExtension installs an optional text-completion adapter.
// Multiple extensions run in registration order; their hooks must be safe for
// concurrent requests and are not applied to chat completions.
func WithCompletionExtension(extension CompletionExtension) ServerOption {
	return func(s *Server) { s.extensions = append(s.extensions, extension) }
}
