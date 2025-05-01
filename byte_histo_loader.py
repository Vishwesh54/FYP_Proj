#!/usr/bin/env python3
import os
import re
import numpy as np

# regex to match exactly two hex digits
HEX_PAIR_RE = re.compile(r'^[0-9A-Fa-f]{2}$')

def extract_byte_histogram(bytes_path):
    """
    Read a BIG-2015 .bytes file, count frequency of each byte (0–255),
    and return a normalized 256-D histogram vector.
    """
    counts = np.zeros(256, dtype=np.int64)
    with open(bytes_path, 'r', errors='ignore') as f:
        for line in f:
            # drop the address, keep only hex pairs
            parts = line.strip().split()[1:]
            for h in parts:
                if HEX_PAIR_RE.match(h):
                    counts[int(h, 16)] += 1
    total = counts.sum()
    return (counts / total) if total>0 else counts.astype(float)

def main(src_root, dst_root):
    """
    Walk src_root for all .bytes files, compute histograms,
    and save each as dst_root/<md5>.npy
    """
    os.makedirs(dst_root, exist_ok=True)  # create output dir :contentReference[oaicite:0]{index=0}
    for dirpath, _, filenames in os.walk(src_root):  # recursive traversal :contentReference[oaicite:1]{index=1}
        for fname in filenames:
            if not fname.lower().endswith('.bytes'):
                continue
            src_path = os.path.join(dirpath, fname)
            md5 = os.path.splitext(fname)[0]
            hist = extract_byte_histogram(src_path)
            out_path = os.path.join(dst_root, f"{md5}.npy")
            # save as binary .npy for fast load later :contentReference[oaicite:2]{index=2}
            np.save(out_path, hist)

if __name__ == '__main__':
    # adjust these paths as needed:
    BYTES_FOLDER = 'Microsoft_dataset/Original/train'      # or 'extracted_test/test'
    OUT_FOLDER   = 'ProcessedData/byte_histograms/'
    main(BYTES_FOLDER, OUT_FOLDER)
