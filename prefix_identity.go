package tinyoai

import "fmt"

// prefixFingerprint returns a stable hexadecimal FNV-1a fingerprint of token
// IDs in prefix order. It identifies a resident context for probability
// records; it is not a security credential.
func prefixFingerprint(ids []int) string {
	h := uint64(14695981039346656037)
	for _, id := range ids {
		h = extendFingerprint(h, id)
	}
	return fmt.Sprintf("%016x", h)
}

// extendFingerprint folds the low four bytes of a token ID into an existing
// FNV-1a state in little-endian order.
func extendFingerprint(h uint64, id int) uint64 {
	for shift := 0; shift < 32; shift += 8 {
		h ^= uint64(byte(id >> shift))
		h *= 1099511628211
	}
	return h
}
