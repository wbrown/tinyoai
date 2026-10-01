package tinyoai

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestStableLMServer compares streamed and ordinary HTTP completions with the
// tiny reference and checks that the repeated prompt reports cache reuse.
func TestStableLMServer(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	r := readReference(t, "testdata/stablelm/expected.json")
	server := NewServer(m)
	for _, stream := range []bool{false, true} {
		request := map[string]any{
			"model": "clio-v1-legacy", "messages": []map[string]string{{"role": "user", "content": r.Generation.Prompt}},
			"temperature": 0, "max_completion_tokens": len(r.Generation.IDs), "stream": stream,
			"stream_options": map[string]bool{"include_usage": true},
		}
		data, _ := json.Marshal(request)
		rec := httptest.NewRecorder()
		server.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(data))))
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body)
		}
		type response struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				Prompt     int `json:"prompt_tokens"`
				Completion int `json:"completion_tokens"`
				Details    struct {
					Cached int `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
			} `json:"usage"`
		}
		var got, finish string
		prompt, completion, cached := 0, 0, 0
		consume := func(data []byte) {
			var out response
			if err := json.Unmarshal(data, &out); err != nil {
				t.Fatal(err)
			}
			for _, c := range out.Choices {
				got += c.Message.Content + c.Delta.Content
				if c.FinishReason != nil {
					finish = *c.FinishReason
				}
			}
			if out.Usage != nil {
				prompt, completion = out.Usage.Prompt, out.Usage.Completion
				cached = out.Usage.Details.Cached
			}
		}
		if stream {
			if rec.Header().Get("Content-Type") != "text/event-stream" {
				t.Fatal("missing SSE header")
			}
			scan := bufio.NewScanner(rec.Body)
			done := false
			for scan.Scan() {
				line := scan.Text()
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				payload := strings.TrimPrefix(line, "data: ")
				if payload == "[DONE]" {
					done = true
					break
				}
				consume([]byte(payload))
			}
			if err := scan.Err(); err != nil {
				t.Fatal(err)
			}
			if !done {
				t.Fatal("missing SSE done")
			}
		} else {
			consume(rec.Body.Bytes())
		}
		if got != r.Generation.Text || prompt != len(m.Encode(r.Generation.Prompt)) || completion != len(r.Generation.IDs) || finish != "length" {
			t.Fatalf("stream=%v: text=%q usage=%d/%d finish=%q", stream, got, prompt, completion, finish)
		}
		wantCached := 0
		if stream {
			wantCached = prompt - 1
		}
		if cached != wantCached {
			t.Fatalf("stream=%v cached=%d, want %d", stream, cached, wantCached)
		}
	}
}
