package tinyoai

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTestServer starts an httptest server backed by the embedded model.
func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	s, err := NewDefaultServer()
	if err != nil {
		t.Fatalf("NewDefaultServer: %v", err)
	}
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	return srv
}

func postJSON(t *testing.T, url string, body map[string]any) *http.Response {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	return resp
}

func TestServeChatCompletion(t *testing.T) {
	srv := newTestServer(t)
	resp := postJSON(t, srv.URL, map[string]any{
		"model": "stories260K",
		"messages": []map[string]any{
			{"role": "user", "content": "Once upon a time"},
		},
		"max_completion_tokens": 24,
		"seed":                  7,
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if out.Object != "chat.completion" {
		t.Errorf("object = %q", out.Object)
	}
	if len(out.Choices) != 1 {
		t.Fatalf("choices = %d", len(out.Choices))
	}
	c := out.Choices[0]
	t.Logf("content=%q finish=%s usage=%d/%d", c.Message.Content, c.FinishReason, out.Usage.PromptTokens, out.Usage.CompletionTokens)
	if c.Message.Role != "assistant" {
		t.Errorf("role = %q", c.Message.Role)
	}
	if strings.TrimSpace(c.Message.Content) == "" {
		t.Error("empty content")
	}
	if c.FinishReason != "length" && c.FinishReason != "stop" {
		t.Errorf("finish_reason = %q", c.FinishReason)
	}
	if out.Usage.PromptTokens == 0 || out.Usage.CompletionTokens == 0 {
		t.Errorf("usage = %+v", out.Usage)
	}
	if out.Usage.TotalTokens != out.Usage.PromptTokens+out.Usage.CompletionTokens {
		t.Errorf("total %d != %d+%d", out.Usage.TotalTokens, out.Usage.PromptTokens, out.Usage.CompletionTokens)
	}
}

func TestServeStreaming(t *testing.T) {
	srv := newTestServer(t)
	resp := postJSON(t, srv.URL, map[string]any{
		"model":                 "stories260K",
		"messages":              []map[string]any{{"role": "user", "content": "The cat"}},
		"max_completion_tokens": 24,
		"seed":                  3,
		"stream":                true,
		"stream_options":        map[string]any{"include_usage": true},
	})
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}

	var content strings.Builder
	var sawRole, sawFinish, sawUsage, sawDone bool
	var completionTokens int

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDone = true
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("bad chunk %q: %v", payload, err)
		}
		if chunk.Usage != nil {
			sawUsage = true
			completionTokens = chunk.Usage.CompletionTokens
		}
		for _, ch := range chunk.Choices {
			if ch.Delta.Role != "" {
				sawRole = true
			}
			content.WriteString(ch.Delta.Content)
			if ch.FinishReason != nil && *ch.FinishReason != "" {
				sawFinish = true
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan: %v", err)
	}

	t.Logf("streamed=%q", content.String())
	if !sawRole {
		t.Error("missing opening role chunk")
	}
	if strings.TrimSpace(content.String()) == "" {
		t.Error("no streamed content")
	}
	if !sawFinish {
		t.Error("missing finish_reason chunk")
	}
	if !sawUsage || completionTokens == 0 {
		t.Error("missing usage chunk")
	}
	if !sawDone {
		t.Error("missing [DONE] sentinel")
	}
}

func TestServeRejectsNonPost(t *testing.T) {
	srv := newTestServer(t)
	resp, err := http.Get(srv.URL + "/v1/chat/completions")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
}
