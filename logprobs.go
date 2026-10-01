package tinyoai

import (
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// TokenProbability describes the unmodified model distribution, before sampling.
type TokenProbability struct {
	// ID is the vocabulary index.
	ID int `json:"id"`
	// Text is a printable decoded piece; special IDs and isolated invalid UTF-8
	// byte fragments have empty text.
	Text string `json:"text"`
	// Logprob is the natural logarithm of probability under the full, unmodified
	// model distribution.
	Logprob float64 `json:"logprob"`
}

// TokenLogprob associates a chosen token and alternatives with context identity
// and UTF-8 byte offsets. An isolated invalid byte fragment has empty Text;
// probability records do not merge fragments the way the streaming decoder does.
type TokenLogprob struct {
	TokenProbability
	// Position is the absolute next-token position, including the backend's BOS
	// position.
	Position int `json:"position"`
	// Prefix fingerprints the preceding token IDs for optional branch expansion;
	// it is not an authorization token.
	Prefix string `json:"prefix"`
	// Start is the inclusive UTF-8 byte offset in the original prompt or raw
	// completion.
	Start int `json:"start"`
	// End is the exclusive byte offset; stop handling can make raw completion
	// spans extend beyond returned text.
	End int `json:"end"`
	// Top contains the retained alternatives, not a renormalized distribution.
	Top []TokenProbability `json:"top"`
}

// ProbabilityEvent groups raw-model probability records delivered during
// prompt preparation or completion.
type ProbabilityEvent struct {
	// Phase is "prompt" or "completion" and identifies which text the byte
	// offsets refer to.
	Phase string `json:"phase"`
	// Tokens contains records in token order for this event.
	Tokens []TokenLogprob `json:"tokens"`
}

type logprobRow struct {
	Chosen float64
	IDs    []int
	Values []float64
}

// summarizeLogits computes the chosen token's log probability and up to count
// alternatives from a nonempty full-vocabulary logit vector. Normalization
// always includes every vocabulary entry; the shortlist only reduces retained
// output. Ties favor the lower token ID.
func summarizeLogits(logits []float32, chosen, count int) logprobRow {
	best := float64(logits[0])
	for _, v := range logits {
		best = math.Max(best, float64(v))
	}
	total := 0.0
	ids := make([]int, 0, count+1)
	for i, v := range logits {
		total += math.Exp(float64(v) - best)
		// IDs arrive in ascending order, so equal scores cannot displace the
		// existing last candidate. Keep the normalization sum unchanged.
		if count > 0 && len(ids) == count && v <= logits[ids[count-1]] {
			continue
		}
		at := sort.Search(len(ids), func(j int) bool { return v > logits[ids[j]] || v == logits[ids[j]] && i < ids[j] })
		if at < count {
			ids = append(ids, i)
			copy(ids[at+1:], ids[at:len(ids)-1])
			ids[at] = i
			if len(ids) > count {
				ids = ids[:count]
			}
		}
	}
	z := best + math.Log(total)
	r := logprobRow{Chosen: float64(logits[chosen]) - z, IDs: ids, Values: make([]float64, len(ids))}
	for i, id := range ids {
		r.Values[i] = float64(logits[id]) - z
	}
	return r
}

// tokenBytes returns a token's decoded bytes before UTF-8 validation or
// special-token filtering. Byte-fallback pieces may therefore produce an
// incomplete UTF-8 sequence.
func (t *nerdstashTokenizer) tokenBytes(id int) string {
	p := t.pieces[id]
	if len(p) == 6 && strings.HasPrefix(p, "<0x") && p[5] == '>' {
		if b, err := strconv.ParseUint(p[3:5], 16, 8); err == nil {
			return string([]byte{byte(b)})
		}
	}
	return strings.ReplaceAll(p, "▁", " ")
}

// probability attaches printable text to an ID and raw-model log probability.
// Special tokens and isolated invalid UTF-8 fragments retain their IDs but
// have empty Text.
func (t *nerdstashTokenizer) probability(id int, p float64) TokenProbability {
	text := t.tokenBytes(id)
	if t.special[id] || !utf8.ValidString(text) {
		text = ""
	}
	return TokenProbability{ID: id, Text: text, Logprob: p}
}

// probabilityRecord combines a chosen-token row with byte offsets and
// deterministically sorted alternatives. The caller fills Position and Prefix
// from the evaluated context.
func (t *nerdstashTokenizer) probabilityRecord(row logprobRow, id, start, end int) TokenLogprob {
	r := TokenLogprob{TokenProbability: t.probability(id, row.Chosen), Start: start, End: end, Top: make([]TokenProbability, len(row.IDs))}
	order := make([]int, len(row.IDs))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		a, b := order[i], order[j]
		return row.Values[a] > row.Values[b] || row.Values[a] == row.Values[b] && row.IDs[a] < row.IDs[b]
	})
	for i, j := range order {
		r.Top[i] = t.probability(row.IDs[j], row.Values[j])
	}
	return r
}

// promptSpans maps retained tokens to UTF-8 byte intervals in the submitted
// prompt. A separately encoded protected prefix maps from the start, and a
// truncated suffix maps backward from the end, so no span crosses discarded
// text. BOS keeps its zero-length interval.
func (t *nerdstashTokenizer) promptSpans(prompt, header string, ids []int, dropped int) [][2]int {
	spans := make([][2]int, len(ids))
	head := len(ids)
	if dropped > 0 {
		head = 1
		if header != "" {
			head = len(t.encode(header))
		}
	}
	pos := 0
	for i := 1; i < head; i++ {
		end := pos + len(t.tokenBytes(ids[i]))
		spans[i] = [2]int{pos, end}
		pos = end
	}
	if head < len(ids) {
		pos = len(prompt)
		for i := len(ids) - 1; i >= head; i-- {
			start := pos - len(t.tokenBytes(ids[i]))
			spans[i] = [2]int{start, pos}
			pos = start
		}
	}
	return spans
}
