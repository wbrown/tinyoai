// Command tinyoai serves the OpenAI-compatible inference API.
package main

import "github.com/wbrown/tinyoai/servercmd"

// main starts the inference server with the default HTTP handlers.
func main() { servercmd.Main(nil) }
