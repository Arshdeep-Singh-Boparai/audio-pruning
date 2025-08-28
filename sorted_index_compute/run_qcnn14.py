#!/usr/bin/env python3
"""
Run (simplified) QCNN14 sorted-index generation.

Usage:
    python run_qcnn14.py <checkpoint.pth> <output_dir> <gm|op|l1>
"""
import sys
from runner_core import run_simple

# Default QCNN14 layers (from your original order)
QCNN14_LAYERS = [
    "conv_block6.conv2","conv_block6.conv1",
    "conv_block5.conv2","conv_block5.conv1",
    "conv_block4.conv2","conv_block4.conv1",
    "conv_block3.conv2","conv_block3.conv1",
    "conv_block2.conv2","conv_block2.conv1",
    "conv_block1.conv2","conv_block1.conv1",
]

def main():
    if len(sys.argv) != 4:
        print("Usage: python run_qcnn14.py <checkpoint.pth> <output_dir> <gm|op|l1>")
        sys.exit(1)
    ckpt, out_dir, method = sys.argv[1], sys.argv[2], sys.argv[3]
    if method not in ("gm","op","l1"):
        print("Method must be one of: gm, op, l1")
        sys.exit(1)
    run_simple(ckpt, out_dir, method, QCNN14_LAYERS)

if __name__ == "__main__":
    main()
