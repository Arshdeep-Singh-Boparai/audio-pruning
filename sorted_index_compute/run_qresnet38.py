#!/usr/bin/env python3
"""
Run (simplified) QResNet38 sorted-index generation.

Usage:
    python run_qresnet38.py <checkpoint.pth> <output_dir> <gm|op|l1>
"""
import sys
from runner_core import run_simple

# Minimal QResNet38 example (extend if needed)
QRESNET38_LAYERS = [
    "conv_block_after1.conv1",
    "conv_block_after1.conv2",
]

def main():
    if len(sys.argv) != 4:
        print("Usage: python run_qresnet38.py <checkpoint.pth> <output_dir> <gm|op|l1>")
        sys.exit(1)
    ckpt, out_dir, method = sys.argv[1], sys.argv[2], sys.argv[3]
    if method not in ("gm","op","l1"):
        print("Method must be one of: gm, op, l1")
        sys.exit(1)
    run_simple(ckpt, out_dir, method, QRESNET38_LAYERS)

if __name__ == "__main__":
    main()
