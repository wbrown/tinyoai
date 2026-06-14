// Command tinyoai runs a pure-Go, OpenAI-compatible /v1/chat/completions server
// backed by the embedded stories260K model. It is useful as a real, offline
// inference backend for tests and local experimentation.
package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/wbrown/tinyoai"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	flag.Parse()

	srv, err := tinyoai.NewDefaultServer()
	if err != nil {
		log.Fatalf("load embedded model: %v", err)
	}
	log.Printf("tinyoai: serving stories260K on %s (POST /v1/chat/completions)", *addr)
	if err := http.ListenAndServe(*addr, srv); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
