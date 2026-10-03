package servercmd

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/wbrown/tinyoai"
)

type adapterModel struct {
	info          tinyoai.LoRAInfo
	loads, scales int
}

// Generate provides the base server interface for adapter endpoint tests.
func (m *adapterModel) Generate(string, tinyoai.GenerateOptions) (tinyoai.GenerateResult, error) {
	return tinyoai.GenerateResult{}, nil
}

// LoadLoRA simulates registered load failures without accessing the filesystem.
func (m *adapterModel) LoadLoRA(_ context.Context, dir string, scale float64) (tinyoai.LoRAInfo, error) {
	if dir == "bad" {
		return tinyoai.LoRAInfo{}, errors.New("invalid adapter")
	}
	m.loads++
	m.info = tinyoai.LoRAInfo{SHA256: dir, Scale: scale}
	return m.info, nil
}

// SetLoRAScale records reuse of already-loaded tensors.
func (m *adapterModel) SetLoRAScale(_ context.Context, scale float64) (tinyoai.LoRAInfo, error) {
	m.scales++
	m.info.Scale = scale
	return m.info, nil
}

// LoRAInfo returns the current mock selection.
func (m *adapterModel) LoRAInfo(context.Context) (tinyoai.LoRAInfo, error) { return m.info, nil }

// TestLoRAEndpoint checks allowlisting, state inspection, scaling without a
// reload, failure atomicity, strict request decoding, and ordinary API routing.
func TestLoRAEndpoint(t *testing.T) {
	m := &adapterModel{}
	h := newLoRAHandler(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(202) }), map[string]tinyoai.Generator{"model": m}, loRAFlags{"test": "weights", "broken": "bad"})
	request := func(method, path, body string) *httptest.ResponseRecorder {
		t.Helper()
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(method, path, strings.NewReader(body)))
		return w
	}
	if request("GET", "/v1/models", "").Code != 202 {
		t.Fatal("lost ordinary API")
	}
	for _, body := range []string{`{}`, `{"model":"model","adapter":"/etc/password"}`, `{"model":"model","adapter":"test","path":"weights"}`, `{"model":"model","adapter":"test"}{}`, `{"model":"model","adapter":"test","scale":"1"}`} {
		if request("POST", "/v1/adapters", body).Code != 400 {
			t.Fatal("accepted invalid request", body)
		}
	}
	if m.loads != 0 {
		t.Fatal("invalid request reached loader")
	}
	if request("POST", "/v1/adapters", `{"model":"model","adapter":"test","scale":0.5}`).Code != 200 || m.loads != 1 || m.info.Scale != 0.5 {
		t.Fatal("load failed")
	}
	if request("POST", "/v1/adapters", `{"model":"model","adapter":"test","scale":0}`).Code != 200 || m.scales != 1 || m.loads != 1 {
		t.Fatal("scale reloaded weights")
	}
	if request("POST", "/v1/adapters", `{"model":"model","adapter":"broken"}`).Code != 400 || h.selected["model"] != "test" {
		t.Fatal("failed load changed selection")
	}
	if body := request("GET", "/v1/adapters", "").Body.String(); !strings.Contains(body, `"sha256":"weights"`) || !strings.Contains(body, `"adapter":"test"`) {
		t.Fatal(body)
	}
	if request("POST", "/v1/adapters", `{"model":"model","adapter":"base"}`).Code != 200 || m.loads != 2 || h.selected["model"] != "base" {
		t.Fatal("unload failed")
	}
	if request("DELETE", "/v1/adapters", "").Code != 405 {
		t.Fatal("accepted wrong method")
	}
	f := loRAFlags{}
	if f.Set("test=./adapter") != nil || f.Set("test=./again") == nil || f.Set("base=./adapter") == nil || f.Set("no separator") == nil {
		t.Fatal("flag validation failed")
	}
}

type waitingAdapterModel struct {
	adapterModel
	entered chan struct{}
}

// LoRAInfo holds the first request as though inference still owns the model gate.
func (m *waitingAdapterModel) LoRAInfo(ctx context.Context) (tinyoai.LoRAInfo, error) {
	m.entered <- struct{}{}
	<-ctx.Done()
	return tinyoai.LoRAInfo{}, ctx.Err()
}

// TestLoRAEndpointQueuedCancellation checks that a cancelled request returns
// before an earlier request releases the handler, without reaching the model.
func TestLoRAEndpointQueuedCancellation(t *testing.T) {
	for _, method := range []string{http.MethodGet, http.MethodPost} {
		t.Run(method, func(t *testing.T) {
			m := &waitingAdapterModel{entered: make(chan struct{}, 1)}
			h := newLoRAHandler(http.NotFoundHandler(), map[string]tinyoai.Generator{"model": m}, loRAFlags{"test": "weights"})
			firstCtx, releaseFirst := context.WithCancel(context.Background())
			firstDone := make(chan struct{})
			var queuedDone chan struct{}
			wait := func(done <-chan struct{}) {
				t.Helper()
				select {
				case <-done:
				case <-time.After(2 * time.Second):
					t.Fatal("adapter request did not finish")
				}
			}
			t.Cleanup(func() {
				releaseFirst()
				wait(firstDone)
				if queuedDone != nil {
					wait(queuedDone)
				}
			})
			go func() {
				defer close(firstDone)
				h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/v1/adapters", nil).WithContext(firstCtx))
			}()
			wait(m.entered)

			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			queuedDone = make(chan struct{})
			w := httptest.NewRecorder()
			go func() {
				defer close(queuedDone)
				body := strings.NewReader(`{"model":"model","adapter":"test"}`)
				h.ServeHTTP(w, httptest.NewRequest(method, "/v1/adapters", body).WithContext(ctx))
			}()
			wait(queuedDone)
			if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), context.Canceled.Error()) {
				t.Fatalf("cancelled request: status %d, body %q", w.Code, w.Body.String())
			}
			if len(m.entered) != 0 || m.loads != 0 || h.selected["model"] != "base" {
				t.Fatal("cancelled request reached the model or changed selection")
			}

			releaseFirst()
			wait(firstDone)
			w = httptest.NewRecorder()
			h.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/adapters", strings.NewReader(`{"model":"model","adapter":"test"}`)))
			if w.Code != http.StatusOK || m.loads != 1 || h.selected["model"] != "test" {
				t.Fatal("cancellation prevented a later adapter selection")
			}
		})
	}
}
