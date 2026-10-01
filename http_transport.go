package tinyoai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

const maxCompletionBody = 4 << 20
const streamWriteTimeout = 30 * time.Second

// readCompletionBody bounds both completion endpoints before JSON decoding or
// tokenization. Extensions receive the same bounded bytes as the base adapter.
func readCompletionBody(w http.ResponseWriter, r *http.Request) ([]byte, error) {
	return io.ReadAll(http.MaxBytesReader(w, r.Body, maxCompletionBody))
}

// eventStream owns SSE framing and transport failure handling for one request.
// A failed stream cancels inference and suppresses subsequent writes. Callbacks
// and the handler use it synchronously on the generation goroutine.
type eventStream struct {
	w          http.ResponseWriter
	controller *http.ResponseController
	cancel     context.CancelFunc
	err        error
}

// newEventStream checks flush support before committing streaming headers.
// It follows the same middleware Unwrap convention as ResponseController.
func newEventStream(w http.ResponseWriter, cancel context.CancelFunc) (*eventStream, error) {
	if !canFlush(w) {
		return nil, fmt.Errorf("streaming unsupported")
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	return &eventStream{w: w, controller: http.NewResponseController(w), cancel: cancel}, nil
}

// canFlush recognizes both Go flush interfaces through transparent middleware.
func canFlush(w http.ResponseWriter) bool {
	for {
		switch v := w.(type) {
		case interface{ FlushError() error }, http.Flusher:
			return true
		case interface{ Unwrap() http.ResponseWriter }:
			w = v.Unwrap()
		default:
			return false
		}
	}
}

// fail records the first transport or serialization error and cancels the
// request, releasing a backend's generation gate at its next cancellation check.
func (s *eventStream) fail(err error) {
	if err != nil && s.err == nil {
		s.err = err
		s.cancel()
	}
}

// send encodes one JSON event. Invalid extension payloads terminate the stream
// instead of emitting an empty, malformed data frame.
func (s *eventStream) send(value any) {
	if s.err != nil {
		return
	}
	data, err := json.Marshal(value)
	if err != nil {
		s.fail(err)
		return
	}
	s.write(string(data))
}

// write emits and flushes one data frame with a fresh write deadline. Test
// recorders and non-network writers may not support deadlines; actual deadline
// errors, write failures, and flush failures cancel inference.
func (s *eventStream) write(data string) {
	if s.err != nil {
		return
	}
	if err := s.controller.SetWriteDeadline(time.Now().Add(streamWriteTimeout)); err != nil && !errors.Is(err, http.ErrNotSupported) {
		s.fail(err)
		return
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", data); err != nil {
		s.fail(err)
		return
	}
	s.fail(s.controller.Flush())
}
