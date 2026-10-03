//go:build mlx && darwin && arm64 && cgo

package tinyoai

import mx "github.com/wbrown/tinyoai/internal/mlx"

// Fork retains weights and evaluated KV with independent handle ownership. MLX
// array updates cannot donate shared buffers, preserving either session when
// the other appends or rewinds. Each session owns its stream and Go layer state.
func (n *nativeMLX) Fork() (mlxForwarder, error) {
	child := &nativeMLX{config: n.config, group: n.group, bits: n.bits,
		kvBits: n.kvBits, offset: n.offset, probabilityBatch: n.probabilityBatch,
		topKBlock: n.topKBlock, reserveTokens: n.reserveTokens,
		fusedProjections: n.fusedProjections, projectionsPacked: n.projectionsPacked,
		densePrefill: n.densePrefill, layerBatch: n.layerBatch, outputRoot: n.outputRoot,
		weights: make(map[string]mx.Array), layers: make([]mlxLayer, len(n.layers))}
	err := mx.Run(func() {
		child.ctx = mx.New()
		owned := make(map[mx.Array]mx.Array, len(n.weights))
		for name, w := range n.weights {
			retained := w.Retain()
			child.weights[name], owned[w] = retained, retained
		}
		if n.lora != nil {
			child.lora = &mlxLoRA{info: n.lora.info, factor: n.lora.factor, pairs: make(map[mx.Array]mlxLoRAPair)}
			for key, pair := range n.lora.pairs {
				retained := mlxLoRAPair{a: pair.a.Retain()}
				child.lora.pairs[owned[key]] = retained
				retained.b = pair.b.Retain()
				child.lora.pairs[owned[key]] = retained
			}
		}
		// All matrix and norm fields borrow handles owned by the weight map.
		weight := func(w mx.Array) mx.Array {
			if !w.Valid() {
				return mx.Array{}
			}
			v, ok := owned[w]
			if !ok {
				mx.Fail("cannot fork an unowned weight handle")
			}
			return v
		}
		matrix := func(w mlxMatrix) mlxMatrix {
			return mlxMatrix{weight(w.weight), weight(w.scales), weight(w.biases)}
		}
		child.embed, child.head = matrix(n.embed), matrix(n.head)
		child.norm, child.bias = weight(n.norm), weight(n.bias)
		for i, l := range n.layers {
			d := &child.layers[i]
			d.norm, d.bias = weight(l.norm), weight(l.bias)
			d.q, d.k, d.v, d.o = matrix(l.q), matrix(l.k), matrix(l.v), matrix(l.o)
			d.gate, d.up, d.down = matrix(l.gate), matrix(l.up), matrix(l.down)
			d.qkv, d.gateUp = matrix(l.qkv), matrix(l.gateUp)
			// Assign each handle as it is retained, allowing Close to clean up
			// a partially constructed fork if the native API fails.
			for _, pair := range [][2]*mlxKV{{&d.keys, &n.layers[i].keys}, {&d.values, &n.layers[i].values}} {
				for _, arrays := range [][2]*mx.Array{{&pair[0].data, &pair[1].data}, {&pair[0].scales, &pair[1].scales}, {&pair[0].biases, &pair[1].biases}} {
					if arrays[1].Valid() {
						*arrays[0] = arrays[1].Retain()
					}
				}
			}
		}
	})
	if err != nil {
		_ = child.Close()
		return nil, err
	}
	return child, nil
}
