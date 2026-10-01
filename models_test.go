package tinyoai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type modelGenerator struct {
	text    string
	wait    chan struct{}
	entered chan struct{}
}

// Generate optionally waits for release or cancellation, then emits a
// model-specific marker for routing and concurrency tests.
func (g *modelGenerator) Generate(_ string, o GenerateOptions) (GenerateResult, error) {
	if g.wait != nil {
		close(g.entered)
		select {
		case <-g.wait:
		case <-o.Context.Done():
			return GenerateResult{}, o.Context.Err()
		}
	}
	if o.OnToken != nil {
		o.OnToken(g.text)
	}
	return GenerateResult{Text: g.text, PromptTokens: 1, CompletionTokens: 1, FinishReason: "length"}, nil
}

// TestNamedModels checks copied registry ownership, sorted discovery, model
// routing in both HTTP APIs, and pre-stream rejection of missing or ambiguous
// model IDs.
func TestNamedModels(t *testing.T) {
	registry := map[string]Generator{"clio-accel": &modelGenerator{text: "GPU"}, "clio-cpu": &modelGenerator{text: "CPU"}}
	s, err := NewModelServer(registry)
	if err != nil {
		t.Fatal(err)
	}
	delete(registry, "clio-cpu") // Caller mutation must not alter the server.
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest("GET", "/v1/models", nil))
	var listing struct {
		Object string
		Data   []struct {
			ID, Object, OwnedBy string
			Created             int64
		}
	}
	if err = json.Unmarshal(w.Body.Bytes(), &listing); err != nil {
		t.Fatal(err)
	}
	if listing.Object != "list" || len(listing.Data) != 2 || listing.Data[0].ID != "clio-accel" || listing.Data[1].ID != "clio-cpu" || listing.Data[0].Object != "model" || listing.Data[0].Created == 0 {
		t.Fatal(w.Body)
	}
	for _, path := range []string{"/v1/completions", "/v1/chat/completions"} {
		for id, want := range map[string]string{"clio-accel": "GPU", "clio-cpu": "CPU"} {
			for _, stream := range []bool{false, true} {
				body := fmt.Sprintf(`{"model":%q,"prompt":"x","messages":[{"role":"user","content":"x"}],"max_tokens":1,"stream":%t}`, id, stream)
				w = httptest.NewRecorder()
				s.ServeHTTP(w, httptest.NewRequest("POST", path, strings.NewReader(body)))
				if w.Code != 200 || !strings.Contains(w.Body.String(), want) || !strings.Contains(w.Body.String(), `"model":"`+id+`"`) {
					t.Fatal(path, id, stream, w.Code, w.Body)
				}
				if stream && !strings.Contains(w.Body.String(), "data: [DONE]") {
					t.Fatal("incomplete stream")
				}
			}
		}
		for _, tc := range []struct {
			model  string
			status int
		}{{"missing", 404}, {"", 400}} {
			body := fmt.Sprintf(`{"model":%q,"prompt":"x","stream":true}`, tc.model)
			w = httptest.NewRecorder()
			s.ServeHTTP(w, httptest.NewRequest("POST", path, strings.NewReader(body)))
			if w.Code != tc.status || strings.Contains(w.Header().Get("Content-Type"), "event-stream") {
				t.Fatal(path, tc, w.Code, w.Body)
			}
			if tc.status == 404 && !strings.Contains(w.Body.String(), `"code":"model_not_found"`) {
				t.Fatal(w.Body)
			}
		}
	}
}

// TestNamedModelsRunIndependently verifies that a blocked request to one model
// does not hold up another registered model.
func TestNamedModelsRunIndependently(t *testing.T) {
	cpu := &modelGenerator{text: "CPU", wait: make(chan struct{}), entered: make(chan struct{})}
	s, _ := NewModelServer(map[string]Generator{"clio-accel": &modelGenerator{text: "GPU"}, "clio-cpu": cpu})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{"model":"clio-cpu","prompt":"x"}`)).WithContext(ctx))
	}()
	select {
	case <-cpu.entered:
	case <-time.After(time.Second):
		t.Fatal("CPU did not start")
	}
	gpuDone := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{"model":"clio-accel","prompt":"x"}`)))
		gpuDone <- w
	}()
	select {
	case w := <-gpuDone:
		if w.Code != 200 {
			t.Fatal(w.Body)
		}
	case <-time.After(time.Second):
		t.Fatal("GPU request blocked on CPU")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("CPU cancellation did not finish")
	}
}

// TestNamedModelRegistrationAndDefault rejects invalid registrations and
// allows an omitted model ID only for a single-model registry.
func TestNamedModelRegistrationAndDefault(t *testing.T) {
	for _, models := range []map[string]Generator{nil, {"": &modelGenerator{}}, {"bad": nil}} {
		if _, err := NewModelServer(models); err == nil {
			t.Fatal("accepted invalid registry")
		}
	}
	s, _ := NewModelServer(map[string]Generator{"clio-cpu": &modelGenerator{text: "CPU"}})
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{"prompt":"x"}`)))
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"model":"clio-cpu"`) {
		t.Fatal(w.Code, w.Body)
	}
}
