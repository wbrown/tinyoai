package tinyoai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"time"
)

// CompletionRequest contains the standard text-completion fields supported by Server.
type CompletionRequest struct {
	// Model selects a registered generator; an empty ID is allowed for a
	// single-model server.
	Model string `json:"model"`
	// Prompt is the untemplated text to continue.
	Prompt string `json:"prompt"`
	// MaxTokens must be positive; DefaultCompletionRequest supplies 40.
	MaxTokens int `json:"max_tokens"`
	// Temperature scales sampling logits; zero chooses greedily.
	Temperature float64 `json:"temperature"`
	// TopP sets nucleus probability mass in (0, 1]; one disables filtering.
	TopP float64 `json:"top_p"`
	// PresencePenalty subtracts a fixed logit penalty for each previously seen
	// token.
	PresencePenalty float64 `json:"presence_penalty"`
	// FrequencyPenalty subtracts a logit penalty for each occurrence in the
	// history.
	FrequencyPenalty float64 `json:"frequency_penalty"`
	// Seed fixes sampling randomness when present; nil requests a fresh seed.
	Seed *int64 `json:"seed"`
	// Stream selects SSE instead of a single JSON response.
	Stream bool `json:"stream"`
	// StreamOptions optionally requests a final usage chunk.
	StreamOptions *streamOptions `json:"stream_options"`
	// Stop accepts a JSON string, string array, or null.
	Stop json.RawMessage `json:"stop"`
	// Logprobs requests up to 64 raw-model alternatives per token; zero disables
	// capture. The backend must support probabilities.
	Logprobs int `json:"logprobs,omitempty"`
}

// DefaultCompletionRequest returns the HTTP text-completion defaults before
// JSON decoding. Starting from these defaults distinguishes omitted controls
// from explicit zero values.
func DefaultCompletionRequest() CompletionRequest {
	return CompletionRequest{Temperature: 1, TopP: 1, MaxTokens: 40}
}

// completions serves the text-completion endpoint, applying registered
// decoders before inference and decorators afterward. Streaming keeps
// probability rows paired with emitted text, cancels on write failure, and
// sends usage only when requested.
func (s *Server) completions(w http.ResponseWriter, r *http.Request) {
	req := DefaultCompletionRequest()
	body, err := readCompletionBody(w, r)
	if err != nil {
		s.writeError(w, 400, "read completion request: "+err.Error())
		return
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	if err := dec.Decode(&req); err != nil {
		s.writeError(w, 400, "invalid completion request: "+err.Error())
		return
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		s.writeError(w, 400, "request must contain one JSON object")
		return
	}
	model, ok := s.SelectModel(w, &req.Model)
	if !ok {
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	opts, err := req.Options(ctx)
	if err != nil {
		s.writeError(w, 400, err.Error())
		return
	}
	for _, extension := range s.extensions {
		if extension.Decode != nil {
			if err := extension.Decode(body, &opts); err != nil {
				s.writeError(w, 400, err.Error())
				return
			}
		}
	}
	id, created := s.id(), time.Now().Unix()
	response := func(text string, finish any, rows []TokenLogprob) map[string]any {
		return map[string]any{"id": id, "object": "text_completion", "created": created, "model": req.Model,
			"choices": []map[string]any{{"index": 0, "text": text, "logprobs": completionLogprobs(rows), "finish_reason": finish}}}
	}
	if !req.Stream {
		result, err := model.Generate(req.Prompt, opts)
		if err != nil {
			s.writeError(w, 400, err.Error())
			return
		}
		out := response(result.Text, result.FinishReason, result.Logprobs)
		out["usage"] = usageJSON(result)
		for _, extension := range s.extensions {
			if extension.Result != nil {
				extension.Result(opts, result, out)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(out)
		return
	}
	stream, err := newEventStream(w, cancel)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	send := stream.send
	send(response("", nil, nil))
	if stream.err != nil {
		return
	}
	var pending []TokenLogprob
	opts.OnLogprobs = func(event ProbabilityEvent) {
		if event.Phase == "completion" {
			pending = append(pending, event.Tokens...)
		}
		for _, extension := range s.extensions {
			if extension.Event != nil {
				if data := extension.Event(event); data != nil {
					send(data)
				}
			}
		}
	}
	opts.OnToken = func(piece string) { send(response(piece, nil, pending)); pending = nil }
	result, err := model.Generate(req.Prompt, opts)
	if err != nil {
		send(map[string]any{"error": map[string]string{"message": err.Error()}})
	} else {
		send(response("", result.FinishReason, pending))
		if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
			out := response("", nil, nil)
			out["choices"] = []any{}
			out["usage"] = usageJSON(result)
			for _, extension := range s.extensions {
				if extension.Result != nil {
					extension.Result(opts, result, out)
				}
			}
			// Probability records have already streamed; retain only usage additions.
			send(map[string]any{"id": id, "object": "text_completion", "created": created, "model": req.Model, "choices": []any{}, "usage": out["usage"]})
		}
	}
	stream.write("[DONE]")
}

// Options validates supported text-completion fields and constructs generation
// settings for HTTP or direct use. It accepts stop as a string or string array
// and chooses a fresh seed when omitted. Backend-specific capability checks
// remain the generator's responsibility.
func (req CompletionRequest) Options(ctx context.Context) (GenerateOptions, error) {
	if req.Logprobs < 0 || req.Logprobs > 64 {
		return GenerateOptions{}, fmt.Errorf("logprobs must be between 0 and 64")
	}
	if req.MaxTokens < 1 || req.Temperature < 0 || req.Temperature > 100 || math.IsNaN(req.Temperature) {
		return GenerateOptions{}, fmt.Errorf("max_tokens must be positive and temperature 0–100")
	}
	sampling := DefaultSampling()
	sampling.TopP, sampling.PresencePenalty, sampling.FrequencyPenalty = req.TopP, req.PresencePenalty, req.FrequencyPenalty
	if err := sampling.Validate(); err != nil {
		return GenerateOptions{}, err
	}
	var stops []string
	if len(req.Stop) > 0 && string(req.Stop) != "null" {
		var one string
		if json.Unmarshal(req.Stop, &one) == nil {
			stops = []string{one}
		} else if json.Unmarshal(req.Stop, &stops) != nil {
			return GenerateOptions{}, fmt.Errorf("stop must be a string or string array")
		}
	}
	seed := rand.Int63()
	if req.Seed != nil {
		seed = *req.Seed
	}
	return GenerateOptions{Context: ctx, Logprobs: req.Logprobs, MaxTokens: req.MaxTokens, Temperature: req.Temperature, Seed: seed, Stop: stops, Sampling: &sampling}, nil
}

// completionLogprobs converts engine rows into the standard text-completion
// logprobs object, or nil when absent. If distinct token IDs decode to the
// same text, the text-keyed alternatives map keeps the highest log
// probability.
func completionLogprobs(rows []TokenLogprob) any {
	if len(rows) == 0 {
		return nil
	}
	tokens := make([]string, 0, len(rows))
	values := make([]float64, 0, len(rows))
	offsets := make([]int, 0, len(rows))
	tops := make([]map[string]float64, 0, len(rows))
	for _, row := range rows {
		tokens = append(tokens, row.Text)
		values = append(values, row.Logprob)
		offsets = append(offsets, row.Start)
		alternatives := make(map[string]float64, len(row.Top))
		for _, token := range row.Top {
			if old, ok := alternatives[token.Text]; !ok || token.Logprob > old {
				alternatives[token.Text] = token.Logprob
			}
		}
		tops = append(tops, alternatives)
	}
	return map[string]any{"tokens": tokens, "token_logprobs": values, "text_offset": offsets, "top_logprobs": tops}
}
