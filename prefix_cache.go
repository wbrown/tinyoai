package tinyoai

// One context per model bounds retained memory and keeps ownership simple.
// Generate's channel allows cancelled requests to leave while waiting.
type stableContext struct {
	state  *stableState
	tokens []int // Only tokens whose forward pass completed successfully.
}

// resize grows the retained context to a power-of-two capacity, capped at the
// model limit. It preserves only completed positions and copies each KV head
// separately because growth changes the head stride. It never shrinks an
// existing context.
func (c *stableContext) resize(m *StableLM, needed int) {
	capacity := 1
	for capacity < needed {
		capacity *= 2
	}
	capacity = min(capacity, m.config.MaxPositionEmbeddings)
	if c.state == nil {
		c.state = m.newState(capacity)
		return
	}
	s := c.state
	heads, head := m.config.NumKeyValueHeads, m.config.HiddenSize/m.config.NumAttentionHeads
	old := len(s.att) / m.config.NumAttentionHeads
	if old >= capacity {
		return
	}
	// Head-major storage has a capacity-dependent stride. Copy each valid
	// head separately when growing; a flat prefix copy would corrupt it.
	growKV(s.keys, old, capacity, heads, head, len(c.tokens))
	growKV(s.values, old, capacity, heads, head, len(c.tokens))
	growKV(s.keys16, old, capacity, heads, head, len(c.tokens))
	growKV(s.values16, old, capacity, heads, head, len(c.tokens))
	if s.keys16 != nil {
		s.keyWorkspace, s.valueWorkspace = make([]float32, capacity*heads*head), make([]float32, capacity*heads*head)
	}
	s.att = make([]float32, capacity*m.config.NumAttentionHeads)
	if s.batch != nil {
		s.batch.att = nil
	}
}
