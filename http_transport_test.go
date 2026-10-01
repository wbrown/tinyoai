package tinyoai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type transportGenerator struct {
	calls, tokens int
	cancelled     bool
}

// Generate emits until the transport cancels, without depending on model weights.
func (g *transportGenerator) Generate(_ string, o GenerateOptions) (GenerateResult, error) {
	g.calls++
	for i := 0; i < 4; i++ {
		if err := o.Context.Err(); err != nil {
			g.cancelled = true
			return GenerateResult{}, err
		}
		g.tokens++
		if o.OnToken != nil {
			o.OnToken("word ")
		}
	}
	return GenerateResult{Text: "word word word word ", FinishReason: "length"}, nil
}

type failingStream struct {
	header                     http.Header
	writes, flushes, deadlines int
	writeFailure, flushFailure int
	deadlineFailure            bool
}

// Header returns the fake writer's mutable response headers.
func (w *failingStream) Header() http.Header { return w.header }

// WriteHeader accepts status without committing any network traffic.
func (w *failingStream) WriteHeader(int) {}

// SetWriteDeadline records deadline coverage and can simulate a controller error.
func (w *failingStream) SetWriteDeadline(deadline time.Time) error {
	w.deadlines++
	if deadline.Before(time.Now()) {
		return fmt.Errorf("expired deadline")
	}
	if w.deadlineFailure {
		return context.DeadlineExceeded
	}
	return nil
}

// Write fails at the configured frame, including the opening event if requested.
func (w *failingStream) Write(p []byte) (int, error) {
	w.writes++
	if w.writes == w.writeFailure {
		return 0, errors.New("broken stream")
	}
	return len(p), nil
}

// FlushError exercises ResponseController's error-reporting flush path.
func (w *failingStream) FlushError() error {
	w.flushes++
	if w.flushes == w.flushFailure {
		return errors.New("flush failed")
	}
	return nil
}

type wrappedStream struct{ http.ResponseWriter }

// Unwrap exposes transport capabilities through ordinary middleware wrappers.
func (w wrappedStream) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// TestCompletionTransportFailures verifies both adapters stop before generation
// on an opening failure, cancel active generation, and set every frame's deadline.
func TestCompletionTransportFailures(t *testing.T) {
	for _, path := range []string{"/v1/completions", "/v1/chat/completions"} {
		for _, kind := range []string{"opening", "write", "flush", "deadline", "healthy"} {
			t.Run(path+"/"+kind, func(t *testing.T) {
				w := &failingStream{header: make(http.Header)}
				switch kind {
				case "opening":
					w.writeFailure = 1
				case "write":
					w.writeFailure = 2
				case "flush":
					w.flushFailure = 2
				case "deadline":
					w.deadlineFailure = true
				}
				g := &transportGenerator{}
				body := `{"prompt":"hello","messages":[{"role":"user","content":"hello"}],"stream":true}`
				NewServer(g).ServeHTTP(wrappedStream{w}, httptest.NewRequest("POST", path, strings.NewReader(body)))
				switch kind {
				case "opening", "deadline":
					if g.calls != 0 {
						t.Fatal("generated after opening transport failure")
					}
				case "write", "flush":
					if !g.cancelled || g.tokens != 1 || w.writes != 2 {
						t.Fatalf("failed stream kept running: generator=%+v writer=%+v", g, w)
					}
				case "healthy":
					if g.tokens != 4 || w.writes != 7 {
						t.Fatalf("incomplete stream: %+v %+v", g, w)
					}
				}
				if !w.deadlineFailure && w.deadlines != w.writes {
					t.Fatalf("writes lack deadlines: %+v", w)
				}
			})
		}
	}
}

// TestCompletionBodyLimit ensures neither adapter admits oversized input to a
// generator and ordinary requests still pass through the shared reader.
func TestCompletionBodyLimit(t *testing.T) {
	for _, path := range []string{"/v1/completions", "/v1/chat/completions"} {
		for _, size := range []int{16, maxCompletionBody + 1} {
			g := &transportGenerator{}
			w := httptest.NewRecorder()
			field := `"prompt":"` + strings.Repeat("x", size) + `"`
			if strings.Contains(path, "/chat/") {
				field = `"messages":[{"role":"user","content":"` + strings.Repeat("x", size) + `"}]`
			}
			NewServer(g).ServeHTTP(w, httptest.NewRequest("POST", path, strings.NewReader("{"+field+"}")))
			if size > maxCompletionBody {
				if w.Code != 400 || g.calls != 0 {
					t.Fatalf("%s admitted oversized body: status=%d calls=%d", path, w.Code, g.calls)
				}
			} else if w.Code != 200 || g.calls != 1 {
				t.Fatalf("%s rejected valid body", path)
			}
		}
	}
}

// TestChatRejectsInvalidGeneration rejects negative controls before delegating
// to a custom generator that may not implement its own validation.
func TestChatRejectsInvalidGeneration(t *testing.T) {
	for _, field := range []string{`"max_tokens":-1`, `"max_completion_tokens":-1`, `"temperature":-1`} {
		g := &transportGenerator{}
		w := httptest.NewRecorder()
		NewServer(g).ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("{"+field+"}")))
		if w.Code != 400 || g.calls != 0 {
			t.Fatalf("accepted %s: status=%d calls=%d", field, w.Code, g.calls)
		}
	}
}

// TestEventSerializationFailure cancels malformed extension events without
// putting an empty JSON data frame on the wire.
func TestEventSerializationFailure(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := httptest.NewRecorder()
	stream, err := newEventStream(w, cancel)
	if err != nil {
		t.Fatal(err)
	}
	stream.send(make(chan int))
	stream.write("[DONE]")
	if ctx.Err() == nil || w.Body.Len() != 0 {
		t.Fatal("serialization failure did not terminate stream")
	}
}

// TestUnsupportedStreaming rejects writers without flush support before
// emitting an event or calling the generator.
func TestUnsupportedStreaming(t *testing.T) {
	for _, path := range []string{"/v1/completions", "/v1/chat/completions"} {
		g, w := &transportGenerator{}, httptest.NewRecorder()
		hidden := struct{ http.ResponseWriter }{w}
		NewServer(g).ServeHTTP(hidden, httptest.NewRequest("POST", path, strings.NewReader(`{"stream":true}`)))
		if w.Code != 500 || g.calls != 0 || strings.Contains(w.Body.String(), "data:") {
			t.Fatalf("%s committed an unsupported stream: %d %s", path, w.Code, w.Body.String())
		}
	}
}
