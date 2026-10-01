package tinyoai

import (
	"reflect"
	"strings"
	"testing"
)

// TestPromptMetadataSurvivesContextTruncation checks that truncation preserves
// a protected prefix and the newest story tokens while short prompts retain
// their original BPE segmentation.
func TestPromptMetadataSurvivesContextTruncation(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	header := "[ Author: Emilie J.Author; Title: The Title; Tags: steamy; Genre: romance ]\n"
	prompt := header + strings.Repeat("The lighthouse stood by the sea. ", 100)
	opts := GenerateOptions{MaxTokens: 8, TruncatePrompt: true, PromptPrefix: header}
	limit := len(m.Encode(header)) + 32
	ids, dropped, err := preparePrompt(prompt, opts, limit, m.Encode)
	if err != nil {
		t.Fatal(err)
	}
	prefix := m.Encode(header)
	if !reflect.DeepEqual(ids[:len(prefix)], prefix) {
		t.Fatal("metadata or BOS was lost")
	}
	full := m.Encode(prompt)
	tail := len(ids) - len(prefix)
	if !reflect.DeepEqual(ids[len(prefix):], full[len(full)-tail:]) {
		t.Fatal("latest story tokens were lost")
	}
	if len(ids)+opts.MaxTokens-1 != limit || dropped != len(full)-len(ids) {
		t.Fatal("wrong context budget")
	}
	// A short prompt retains its exact original BPE segmentation.
	short := header + "A story."
	ids, dropped, err = preparePrompt(short, opts, 8192, m.Encode)
	if err != nil || dropped != 0 || !reflect.DeepEqual(ids, m.Encode(short)) {
		t.Fatal("short prompt was changed")
	}
	if _, _, err = preparePrompt(prompt, opts, 8, m.Encode); err == nil {
		t.Fatal("oversized metadata was not rejected")
	}
	if _, _, err = preparePrompt("wrong prefix", opts, 8192, m.Encode); err == nil {
		t.Fatal("mismatched prefix was not rejected")
	}
}
