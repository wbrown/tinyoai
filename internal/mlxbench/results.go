package mlxbench

import (
	"encoding/json"
	"math"
	"os"
)

// WriteJSON writes an indented report through a sibling temporary
// file and rename, so readers do not observe a partially written final report.
func WriteJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(path+".tmp", b, 0600); err != nil {
		return err
	}
	return os.Rename(path+".tmp", path)
}

// Argmax returns the first index attaining the largest value in a
// nonempty logit vector.
func Argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// KL computes KL(softmax(p) || softmax(q)) in float64 over equally
// sized, nonempty vocabularies. Separate log-sum-exp normalizers avoid
// overflow; tiny negative roundoff is clamped to zero.
func KL(p, q []float32) float64 {
	pm, qm := float64(p[Argmax(p)]), float64(q[Argmax(q)])
	ps, qs := 0.0, 0.0
	for i := range p {
		ps += math.Exp(float64(p[i]) - pm)
		qs += math.Exp(float64(q[i]) - qm)
	}
	pl, ql := pm+math.Log(ps), qm+math.Log(qs)
	kl := 0.0
	for i := range p {
		lp, lq := float64(p[i])-pl, float64(q[i])-ql
		kl += math.Exp(lp) * (lp - lq)
	}
	return max(0, kl)
}
