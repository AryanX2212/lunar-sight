"""Convert a PyTorch Zero-DCE checkpoint into the .npz this server loads.

Run this once, on a machine that has PyTorch. The server itself never imports
torch: it does inference in NumPy, so the runtime stays small.

    pip install torch
    python tools/convert_weights.py --checkpoint Epoch99.pth --out models/zero_dce_weights.npz

Checkpoints from the reference implementation (Li-Chongyi/Zero-DCE) store the
seven conv layers as e_conv1 ... e_conv7. Keys are normalised here, so a
checkpoint saved from a DataParallel wrapper ('module.' prefixed) also works.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

EXPECTED = ("e_conv1", "e_conv2", "e_conv3", "e_conv4", "e_conv5", "e_conv6", "e_conv7")


def normalise_key(key: str) -> str:
    for prefix in ("module.", "model.", "net."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to the .pth file")
    parser.add_argument("--out", default=os.path.join("models", "zero_dce_weights.npz"))
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch is required for conversion only. pip install torch", file=sys.stderr)
        return 2

    state = torch.load(args.checkpoint, map_location="cpu")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    for wrapper in ("state_dict", "model", "net"):
        if isinstance(state, dict) and wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]
            break

    arrays: dict[str, np.ndarray] = {}
    for raw_key, tensor in state.items():
        key = normalise_key(str(raw_key))
        if key.split(".")[0] in EXPECTED:
            arrays[key] = tensor.detach().cpu().numpy().astype(np.float32)

    missing = [
        f"{layer}.{part}"
        for layer in EXPECTED for part in ("weight", "bias")
        if f"{layer}.{part}" not in arrays
    ]
    if missing:
        print("Checkpoint is missing expected tensors:", file=sys.stderr)
        for key in missing:
            print("  " + key, file=sys.stderr)
        print("\nFound instead:", file=sys.stderr)
        for key in sorted(normalise_key(str(k)) for k in state):
            print("  " + key, file=sys.stderr)
        return 1

    final = arrays["e_conv7.weight"]
    if final.ndim != 4 or final.shape[0] % 3 != 0:
        print(f"Unexpected final layer shape {final.shape}; expected (3n, C, 3, 3).",
              file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    print(f"Wrote {args.out}")
    print(f"  {len(arrays)} tensors, {final.shape[0] // 3} curve iterations")
    print(f"  {sum(a.size for a in arrays.values()):,} parameters")
    print("Restart the server; /api/health will report the weights as loaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
