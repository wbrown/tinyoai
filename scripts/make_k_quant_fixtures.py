"""Independent synthetic block fixtures; requires numpy and gguf==0.17.1."""
from pathlib import Path
import numpy as np
import gguf

output = Path(__file__).resolve().parents[1] / 'testdata' / 'gguf'
output.mkdir(exist_ok=True)
rng = np.random.default_rng(9041)
# Arbitrary packed bits cover every subfield independently of tinyoai's decoder.
# Replace half-float scale fields with finite values so error comparisons stay
# meaningful; include zero, a subnormal, and both signs to exercise edge cases.
for dtype, size, width in [('Q5_K', 176, 256), ('Q6_K', 210, 256), ('Q5_1', 24, 32)]:
    raw = rng.integers(0, 256, size=(8192 // width, size), dtype=np.uint8)
    scales = np.array([0, 2**-24, .013, -.025, .1, -.2, .5, 1], dtype=np.float16)
    for block in range(len(raw)):
        scale = scales[block % len(scales)].tobytes()
        if dtype == 'Q6_K':
            raw[block, 208:] = np.frombuffer(scale, dtype=np.uint8)
        else:
            raw[block, :2] = np.frombuffer(scale, dtype=np.uint8)
            raw[block, 2:4] = np.frombuffer(scales[(block+3) % len(scales)].tobytes(), dtype=np.uint8)
    decoded = gguf.dequantize(raw, getattr(gguf.GGMLQuantizationType, dtype))
    (output / (dtype.lower() + '.bin')).write_bytes(raw.tobytes())
    (output / (dtype.lower() + '-f32.bin')).write_bytes(decoded.astype('<f4').tobytes())
