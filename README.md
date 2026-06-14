# tinyoai

A pure-Go, OpenAI-compatible `/v1/chat/completions` server backed by a tiny
embedded language model. No CGO, no network, no API key, no external
dependencies — just the standard library and a ~1MB embedded model.

It exists to be a **real** inference backend for tests of OpenAI-compatible
clients (such as `github.com/wbrown/openai`): the client talks HTTP to a server
that runs a real transformer forward pass and returns real token counts, finish
reasons, and SSE streams — not a hand-mocked response.

## The model

The embedded default is Andrej Karpathy's **stories260K** — a ~260K-parameter
Llama 2 (8 query heads, 4 key/value heads via grouped-query attention) trained
on the TinyStories dataset and published expressly as a unit-test model. The
weights (`assets/stories260K.bin`, ~1MB) and tokenizer (`assets/tok512.bin`,
~6KB) are embedded with `go:embed`. Output is real but small-model-quality
TinyStories prose.

## Library use

```go
m, err := tinyoai.Default() // embedded stories260K
if err != nil {
    log.Fatal(err)
}

res, err := m.Generate("Once upon a time", tinyoai.GenerateOptions{
    MaxTokens:   64,
    Temperature: 0.8,
    Seed:        1, // reproducible
})
fmt.Println(res.Text, res.FinishReason, res.PromptTokens, res.CompletionTokens)
```

Load your own legacy llama2.c checkpoint with `LoadModel(checkpoint, tokenizer)`.

## As a server

In tests:

```go
s, err := tinyoai.NewDefaultServer()
if err != nil {
    t.Fatal(err)
}
srv := httptest.NewServer(s)
conv.SetEndpoint(srv.URL + "/v1/chat/completions")
```

Or as a standalone binary:

```bash
go run ./cmd/tinyoai -addr :8080
curl localhost:8080/v1/chat/completions -d '{
  "model": "stories260K",
  "messages": [{"role": "user", "content": "Once upon a time"}],
  "max_completion_tokens": 64
}'
```

The server supports streaming (`"stream": true`) with OpenAI-style
`chat.completion.chunk` SSE events and a trailing usage chunk when
`stream_options.include_usage` is set.

## Scope

stories260K is a plain text model — it has no chat/role training, no tool
calling, and no vision. The server flattens message text into a single prompt
and drops roles. It is intended for exercising client wire-protocol behavior
(requests, responses, streaming, token accounting), not for response quality.

## License

MIT. Bundles MIT-licensed material from `tmc/go-llama2`, `karpathy/llama2.c`,
and `karpathy/tinyllamas` — see `LICENSE`.
