# Reading and maintaining tinyoai

Start with [`doc.go`](../doc.go), also available through `go doc .`. It explains inference from tokenization through serving and identifies the source files for each stage.

The documentation follows Knuth's [literate programming](https://cs-faculty.stanford.edu/~knuth/lp.html): explain how the program works and why it is written that way. Source remains ordinary Go, with declaration comments following [Go's conventions](https://go.dev/doc/comment).

## Declaration comments

Give every named function and method a comment beginning with its name, including helpers, tests, benchmarks, and build-specific stubs. Describe its purpose, effects, results, and preconditions. Document memory ownership, aliasing, locks, zero values, and cancellation where relevant.

Exported types, fields, and interfaces must describe their contracts. Use inline comments for non-obvious calculations and invariants: rotary rounding, tensor layout, masking, cache strides, native lifetimes, and probability alignment.

Keep claims about formats, errors, and concurrency consistent with the code. Document engine behavior and supported checkpoints here; consumer-specific workflows belong in the consumer's documentation. Keep each Markdown paragraph on one source line and separate topics with blank lines.

Shared request validation and prefix-reuse policy live in `generation_policy.go`; HTTP limits and SSE delivery live in `http_transport.go`. Architecture-specific forward passes remain explicit. `internal/mlxbench` owns diagnostic scheduling, result identity, and persistence, while its engine adapter keeps native arrays private.

Test comments state the behavior checked and identify independent references. Distinguish correctness assertions from performance measurements. Explain arithmetic differences that prevent bitwise equality. Preserve provenance and license notices.

## Checks

From the repository root:

```sh
go run ./scripts/check-docs.go
go test ./...
go vet ./...
```

The documentation checker parses all Go files, including tests and inactive build variants. It checks comment coverage and name prefixes without model weights or a native SDK. Review comment accuracy separately.

For SIMD checks, use the toolchain pinned in the README. For native builds, follow [`mlx-native.md`](mlx-native.md). Default tests skip cases requiring model paths or Metal opt-in.

Documentation edits do not require full-checkpoint performance runs. Changes to arithmetic, cache layout, or evaluation shape require independent parity checks, including the final supported context positions.

## API limits

Model loaders expect trusted local checkpoints. Format checks reject incompatible assets, but declared dimensions determine allocation sizes; arbitrary uploads can exhaust memory.

The legacy chat adapter ignores `top_p` and tools. Text completions support nucleus sampling, presence/frequency penalties, and extension hooks. Document which backends implement each optional generation field.
