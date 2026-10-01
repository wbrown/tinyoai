// Package tokenizer implements Clio tokenization without inference or platform dependencies.
package tokenizer

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"
)

type addedToken struct {
	ID         int    `json:"id"`
	Content    string `json:"content"`
	Normalized bool   `json:"normalized"`
	Special    bool   `json:"special"`
	SingleWord bool   `json:"single_word"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
}

type mergePair struct{ left, right int }
type mergeRule struct{ rank, id int }

// Tokenizer implements the precise pipeline shipped with Clio:
// added-token extraction, space replacement, ranked BPE, and byte fallback.
// It deliberately rejects other Hugging Face tokenizer pipelines.
// Tokenizer is immutable after Parse and safe for concurrent encoding.
// Vocab, Pieces and Special are read-only views used by inference decoders.
type Tokenizer struct {
	source []byte
	// Vocab maps token pieces to IDs and must be treated as read-only.
	Vocab map[string]int
	// Pieces indexes raw vocabulary pieces by ID; unused padded IDs have empty
	// pieces. Treat the slice as read-only.
	Pieces                    []string
	rules                     map[mergePair]mergeRule
	rawAdded, normalizedAdded []addedToken
	// Special marks IDs omitted by ordinary text decoding. Treat the map as
	// read-only.
	Special map[int]bool
	bos     int
}

// Parse validates the supported Nerdstash byte-fallback BPE pipeline and
// returns an encoder safe for concurrent use. It checks vocabulary IDs, byte
// coverage, merges, added-token rules, normalization, and the BOS template.
// The source bytes are copied; returned vocabulary views must remain
// read-only.
func Parse(data []byte, vocabSize, bos int) (*Tokenizer, error) {
	if vocabSize < 1 || vocabSize > 1<<20 || bos < 0 || bos >= vocabSize {
		return nil, fmt.Errorf("invalid tokenizer vocabulary size or BOS")
	}
	var j struct {
		Added         []addedToken    `json:"added_tokens"`
		Normalizer    json.RawMessage `json:"normalizer"`
		PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
		PostProcessor json.RawMessage `json:"post_processor"`
		Decoder       json.RawMessage `json:"decoder"`
		Model         struct {
			Type         string            `json:"type"`
			Vocab        map[string]int    `json:"vocab"`
			Merges       []json.RawMessage `json:"merges"`
			ByteFallback bool              `json:"byte_fallback"`
			Dropout      *float64          `json:"dropout"`
			UnkToken     string            `json:"unk_token"`
			Prefix       string            `json:"continuing_subword_prefix"`
			Suffix       string            `json:"end_of_word_suffix"`
			IgnoreMerges bool              `json:"ignore_merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &j); err != nil {
		return nil, err
	}
	if j.Model.Type != "BPE" || !j.Model.ByteFallback || j.Model.IgnoreMerges ||
		(j.Model.Dropout != nil && *j.Model.Dropout != 0) || j.Model.Prefix != "" || j.Model.Suffix != "" {
		return nil, fmt.Errorf("unsupported tokenizer model; expected Nerdstash byte-fallback BPE")
	}
	if string(j.PreTokenizer) != "null" && len(j.PreTokenizer) != 0 {
		return nil, fmt.Errorf("unsupported tokenizer pre_tokenizer")
	}
	// Semantic JSON comparisons ignore insignificant whitespace and key order.
	if !sameJSON(j.Normalizer, `{"type":"Sequence","normalizers":[{"type":"Replace","pattern":{"String":" "},"content":"▁"}]}`) ||
		!sameJSON(j.Decoder, `{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]}`) {
		return nil, fmt.Errorf("unsupported tokenizer normalization or decoder pipeline")
	}
	var post struct {
		Type    string            `json:"type"`
		Single  []json.RawMessage `json:"single"`
		Special map[string]struct {
			IDs []int `json:"ids"`
		} `json:"special_tokens"`
	}
	if err := json.Unmarshal(j.PostProcessor, &post); err != nil {
		return nil, err
	}
	var first struct {
		Special *struct {
			ID string `json:"id"`
		} `json:"SpecialToken"`
	}
	if post.Type != "TemplateProcessing" || len(post.Single) != 2 {
		return nil, fmt.Errorf("unsupported tokenizer post_processor")
	}
	if err := json.Unmarshal(post.Single[0], &first); err != nil || first.Special == nil {
		return nil, fmt.Errorf("tokenizer must prepend BOS")
	}
	ids := post.Special[first.Special.ID].IDs
	if len(ids) != 1 || ids[0] != bos || !sameJSON(post.Single[1], `{"Sequence":{"id":"A","type_id":0}}`) {
		return nil, fmt.Errorf("tokenizer BOS/template does not match model")
	}
	t := &Tokenizer{source: append([]byte(nil), data...), Vocab: j.Model.Vocab, Pieces: make([]string, vocabSize), rules: make(map[mergePair]mergeRule), Special: make(map[int]bool), bos: bos}
	if _, ok := t.Vocab[j.Model.UnkToken]; !ok {
		return nil, fmt.Errorf("unknown token missing from vocabulary")
	}
	seen := make(map[int]bool)
	for piece, id := range t.Vocab {
		if id < 0 || id >= vocabSize || seen[id] || piece == "" {
			return nil, fmt.Errorf("invalid vocabulary entry %q: %d", piece, id)
		}
		seen[id] = true
		t.Pieces[id] = piece
	}
	if t.Vocab[first.Special.ID] != bos || !seen[bos] {
		return nil, fmt.Errorf("BOS missing from vocabulary")
	}
	for b := 0; b < 256; b++ {
		if _, ok := t.Vocab[fmt.Sprintf("<0x%02X>", b)]; !ok {
			return nil, fmt.Errorf("missing byte fallback token %02X", b)
		}
	}
	for _, a := range j.Added {
		id, exists := t.Vocab[a.Content]
		if !exists || id != a.ID || a.SingleWord || a.LStrip || a.RStrip {
			return nil, fmt.Errorf("unsupported added token %q", a.Content)
		}
		if a.Special {
			t.Special[a.ID] = true
		}
		if a.Normalized {
			t.normalizedAdded = append(t.normalizedAdded, a)
		} else {
			t.rawAdded = append(t.rawAdded, a)
		}
	}
	for rank, raw := range j.Model.Merges {
		var pair []string
		if err := json.Unmarshal(raw, &pair); err != nil {
			var old string
			if err := json.Unmarshal(raw, &old); err != nil {
				return nil, fmt.Errorf("invalid merge %d", rank)
			}
			pair = strings.Split(old, " ")
		}
		if len(pair) != 2 {
			return nil, fmt.Errorf("invalid merge %d", rank)
		}
		left, lok := t.Vocab[pair[0]]
		right, rok := t.Vocab[pair[1]]
		id, iok := t.Vocab[pair[0]+pair[1]]
		if !lok || !rok || !iok {
			return nil, fmt.Errorf("merge %d references absent vocabulary", rank)
		}
		t.rules[mergePair{left, right}] = mergeRule{rank, id}
	}
	return t, nil
}

// sameJSON compares decoded JSON structure rather than serialization order or
// whitespace. Either invalid input makes the comparison false.
func sameJSON(raw []byte, expected string) bool {
	var a, b any
	if json.Unmarshal(raw, &a) != nil || json.Unmarshal([]byte(expected), &b) != nil {
		return false
	}
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}

// Encode returns fresh token IDs beginning with BOS. Raw added tokens are
// isolated before space normalization; normalized added tokens are isolated
// before ranked BPE, preserving the checkpoint's tokenizer pipeline.
func (t *Tokenizer) Encode(text string) []int {
	ids := []int{t.bos}
	// Non-normalized special tokens are isolated before normalization. The
	// remaining added tokens match normalized text, with longest match winning.
	normalized := func(s string) {
		s = strings.ReplaceAll(s, " ", "▁")
		splitAdded(s, t.normalizedAdded, func(s string) { ids = append(ids, t.bpe(s)...) }, func(id int) { ids = append(ids, id) })
	}
	splitAdded(text, t.rawAdded, normalized, func(id int) { ids = append(ids, id) })
	return ids
}

// splitAdded visits plain spans and added-token IDs in source order. At each
// rune boundary the longest matching added token wins, preventing a shorter
// special token from consuming a longer one's prefix.
func splitAdded(s string, added []addedToken, plain func(string), token func(int)) {
	start := 0
	for i := 0; i < len(s); {
		length, id := 0, 0
		for _, a := range added {
			if len(a.Content) > length && strings.HasPrefix(s[i:], a.Content) {
				length, id = len(a.Content), a.ID
			}
		}
		if length == 0 {
			_, n := utf8.DecodeRuneInString(s[i:])
			i += n
			continue
		}
		if start < i {
			plain(s[start:i])
		}
		token(id)
		i += length
		start = i
	}
	if start < len(s) {
		plain(s[start:])
	}
}

type bpeNode struct {
	id, prev, next int
	alive          bool
}
type mergeCandidate struct{ rank, pos, right, leftID, rightID, id int }
type mergeHeap []mergeCandidate

// Len returns the number of queued merge candidates for container/heap.
func (h mergeHeap) Len() int { return len(h) }

// Less orders candidates by merge rank, then by original left position. This
// reproduces left-to-right selection when merge priorities tie.
func (h mergeHeap) Less(i, j int) bool {
	if h[i].rank == h[j].rank {
		return h[i].pos < h[j].pos
	}
	return h[i].rank < h[j].rank
}

// Swap exchanges heap entries without changing their original token positions.
func (h mergeHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

// Push appends a mergeCandidate supplied by container/heap; heap.Push restores
// ordering after this method returns.
func (h *mergeHeap) Push(x any) { *h = append(*h, x.(mergeCandidate)) }

// Pop removes the final entry after container/heap has moved its selected
// minimum there.
func (h *mergeHeap) Pop() any { a := *h; x := a[len(a)-1]; *h = a[:len(a)-1]; return x }

// bpe encodes one normalized plain span, falling back to UTF-8 bytes for
// unknown runes. A linked sequence of live nodes and a rank heap avoids
// rescanning every pair after each merge. Heap entries snapshot their endpoint
// IDs; stale candidates are discarded before they can merge nodes that have
// since changed.
func (t *Tokenizer) bpe(s string) []int {
	var ids []int
	for _, r := range s {
		if id, ok := t.Vocab[string(r)]; ok {
			ids = append(ids, id)
		} else {
			for _, b := range []byte(string(r)) {
				ids = append(ids, t.Vocab[fmt.Sprintf("<0x%02X>", b)])
			}
		}
	}
	if len(ids) < 2 {
		return ids
	}
	nodes := make([]bpeNode, len(ids))
	for i, id := range ids {
		nodes[i] = bpeNode{id: id, prev: i - 1, next: i + 1, alive: true}
	}
	nodes[len(nodes)-1].next = -1
	h := &mergeHeap{}
	push := func(i int) {
		if i < 0 || nodes[i].next < 0 {
			return
		}
		j := nodes[i].next
		if rule, ok := t.rules[mergePair{nodes[i].id, nodes[j].id}]; ok {
			heap.Push(h, mergeCandidate{rule.rank, i, j, nodes[i].id, nodes[j].id, rule.id})
		}
	}
	for i := range nodes {
		push(i)
	}
	for h.Len() > 0 {
		c := heap.Pop(h).(mergeCandidate)
		l, r := &nodes[c.pos], &nodes[c.right]
		if !l.alive || !r.alive || l.next != c.right || l.id != c.leftID || r.id != c.rightID {
			continue
		}
		l.id = c.id
		l.next = r.next
		r.alive = false
		if r.next >= 0 {
			nodes[r.next].prev = c.pos
		}
		push(l.prev)
		push(c.pos)
	}
	ids = ids[:0]
	for i := 0; i >= 0; i = nodes[i].next {
		ids = append(ids, nodes[i].id)
	}
	return ids
}

// Source lends the original tokenizer JSON for metadata consumers. The returned
// bytes share the tokenizer's immutable storage and must not be modified.
func (t *Tokenizer) Source() []byte { return t.source }
