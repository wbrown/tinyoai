package tinyoai

// attendBatch shares each key/value load across four neighboring queries.
// Every query still attends to its exact causal prefix. The scalar build and
// short tails use the one-query implementation.
func (m *StableLM) attendBatch(l, pos, n int, s *stableState, b *stableBatch) {
	c := m.config
	d, head := c.HiddenSize, c.HiddenSize/c.NumAttentionHeads
	capacity := len(s.att) / c.NumAttentionHeads
	groups := (n + attentionBatchWidth() - 1) / attentionBatchWidth()
	layerKeys, layerValues := s.kvLayer(l)
	parallelRows(groups*c.NumAttentionHeads, n*d*(pos+n), func(begin, end int) {
		view := *s
		for task := begin; task < end; task++ {
			h, j := task/groups, task%groups*attentionBatchWidth()
			count := min(attentionBatchWidth(), n-j)
			if count != 4 {
				for q := j; q < j+count; q++ {
					view.q = b.q[q*d : (q+1)*d]
					view.attOut = b.attOut[q*d : (q+1)*d]
					view.att = b.att[q*len(s.att) : (q+1)*len(s.att)]
					m.attendHeads(l, pos+q, &view, h, h+1)
				}
				continue
			}
			offset := h / (c.NumAttentionHeads / c.NumKeyValueHeads) * capacity * head
			keys := layerKeys[offset : offset+(pos+j+4)*head]
			values := layerValues[offset : offset+(pos+j+4)*head]
			q := b.q[j*d+h*head:]
			scores := b.att[j*len(s.att)+h*capacity:]
			attentionScores4(q, keys, scores, d, len(s.att), head, pos+j+1)
			for i := 0; i < 4; i++ {
				att := scores[i*len(s.att) : i*len(s.att)+pos+j+4]
				// The common prefix was computed together. Only six additional
				// scores remain in the four-query causal triangle.
				for t := pos + j + 1; t <= pos+j+i; t++ {
					att[t] = attentionDot(q[i*d:i*d+head], keys[t*head:(t+1)*head]) / attentionScale(head)
				}
				softmax(att[:pos+j+i+1])
				clear(att[pos+j+i+1:])
			}
			attentionValues4(b.attOut[j*d+h*head:], values, scores, d, len(s.att), head, pos+j+4)
		}
	})
}
