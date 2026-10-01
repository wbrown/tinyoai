"""Recheck saved parity data without loading model weights or running inference.

python scripts/compare_saved_logits.py /path/to/greedy-case
python scripts/compare_saved_logits.py /path/to/greedy-case --go-logits go-simd-logits.safetensors

Requires numpy and safetensors (already used by the reference environment).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def compare(reference, actual, generated_ids):
    """Compare equally shaped float32 traces over identical saved token histories.

    Validate reference argmax IDs, then report full-vocabulary KL and logit error
    using float64 arithmetic that can resolve tiny distribution differences."""
    if reference.shape != actual.shape or reference.ndim != 2:
        raise ValueError(f"incompatible shapes: {reference.shape}, {actual.shape}")
    if reference.shape != (len(generated_ids), 65536):
        raise ValueError("expected one 65,536-way distribution per generated ID")
    if reference.dtype != np.float32 or actual.dtype != np.float32:
        raise ValueError("expected float32 logits")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("non-finite logits")
    if not np.array_equal(reference.argmax(axis=1), generated_ids):
        raise ValueError("reference logits disagree with saved output IDs")

    per_step, max_error, squared_error = [], 0.0, 0.0
    for r32, a32 in zip(reference, actual):
        r, a = r32.astype(np.float64), a32.astype(np.float64)
        p = np.exp(r - r.max())
        p /= p.sum()
        delta = a - r
        # KL(p || softmax(r+delta)) = log(Ep[exp(delta)]) - Ep[delta].
        # Centering delta and using expm1/log1p resolves very small KL without
        # subtracting two nearly equal log-softmax values. This independently
        # checks the Go test's compensated log-softmax implementation.
        centered = delta - np.sum(p * delta)
        kl = float(np.log1p(np.sum(p * np.expm1(centered))))
        if -1e-15 < kl < 0:
            kl = 0.0
        per_step.append(kl)
        max_error = max(max_error, float(np.abs(delta).max()))
        squared_error += float(np.sum(delta * delta))
    matches = actual.argmax(axis=1) == np.asarray(generated_ids)
    return {
        "rows": len(generated_ids), "vocab_size": 65536,
        "argmax_matches": int(matches.sum()),
        "mismatch_steps": np.flatnonzero(~matches).tolist(),
        "max_kl": max(per_step), "mean_kl": float(np.mean(per_step)),
        "per_step_kl": per_step,
        "max_logit_error": max_error,
        "rms_logit_error": float(np.sqrt(squared_error / reference.size)),
    }


def main():
    """Audit local trace files, optionally save a report, and fail on parity violations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--go-logits", default="go-logits.safetensors")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-kl", type=float, default=1e-7)
    parser.add_argument("--allow-argmax-mismatch", action="store_true",
                        help="report argmax drift for a teacher-forced lossy-precision comparison")
    args = parser.parse_args()
    metadata = json.loads((args.case_dir / "reference.json").read_text())
    if metadata["execution_mode"] != "cpu-float32-cached-token":
        parser.error("reference must use CPU float32 cached one-token evaluation")
    reference = load_file(str(args.case_dir / "logits.safetensors"))["logits"]
    actual = load_file(str(args.case_dir / args.go_logits))["logits"]
    result = compare(reference, actual, metadata["generated_ids"])
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_step_kl"}, indent=2))
    if (result["mismatch_steps"] and not args.allow_argmax_mismatch) or not all(0 <= v <= args.max_kl for v in result["per_step_kl"]):
        raise SystemExit("saved logits fail greedy/KL parity")


if __name__ == "__main__":
    main()
