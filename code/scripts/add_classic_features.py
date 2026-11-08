"""
Add handcrafted per-residue features to NPZ files.

This script augments existing NPZ files with classic_feats:
- One-hot encoding (20 dims)
- BLOSUM62 row (20 dims)
- Z-scale descriptors (5 dims)
Total: 45 dimensions per residue

Usage:
    python scripts/add_classic_features.py --task hemo
    python scripts/add_classic_features.py --task sol
    python scripts/add_classic_features.py --task nf
"""
import argparse
import os
import numpy as np
from typing import Dict, List

# ===== Token ID mapping (consistent with graph_data_builder.py) =====
PAD_ID = 0
CLS_ID = 2
SEP_ID = 3
MASK_ID = 4
SPECIAL_IDS = {PAD_ID, CLS_ID, SEP_ID, MASK_ID}

# Standard 20 amino acids (alphabetical order)
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Token ID to amino acid mapping for Rostlab/prot_bert vocabulary
# Based on the tokenizer: L=5, A=6, G=7, V=8, S=9, E=10, R=11, T=12, I=13, D=14,
# P=15, K=16, Q=17, N=18, F=19, Y=20, M=21, H=22, W=23, C=24
TOKEN_TO_AA = {
    5: 'L', 6: 'A', 7: 'G', 8: 'V', 9: 'S',
    10: 'E', 11: 'R', 12: 'T', 13: 'I', 14: 'D',
    15: 'P', 16: 'K', 17: 'Q', 18: 'N', 19: 'F',
    20: 'Y', 21: 'M', 22: 'H', 23: 'W', 24: 'C'
}

# Amino acid to index (for one-hot encoding)
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_LIST)}

# BLOSUM62 matrix (20x20) - rows and columns follow AA_LIST order
BLOSUM62 = np.array([
    # A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    [ 4, -1, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],  # A
    [-1,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],  # C
    [-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],  # D
    [-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],  # E
    [-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],  # F
    [ 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],  # G
    [-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],  # H
    [-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],  # I
    [-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],  # K
    [-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],  # L
    [-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],  # M
    [-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],  # N
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],  # P
    [-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],  # Q
    [-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],  # R
    [ 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],  # S
    [ 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],  # T
    [ 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],  # V
    [-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],  # W
    [-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],  # Y
], dtype=np.float32)

# Z-scale descriptors (20 x 5) - rows follow AA_LIST order
# Source: Sandberg et al. (1998) principal component analysis
ZSCALE = np.array([
    # z1     z2     z3     z4     z5
    [ 0.07, -1.73,  0.09, -0.16,  0.91],  # A
    [ 0.71, -0.97,  4.13,  0.18, -0.20],  # C
    [ 3.64,  1.13,  2.36, -0.37,  0.82],  # D
    [ 3.08,  0.39, -0.07,  1.09,  0.86],  # E
    [-4.92,  1.30,  0.45, -0.32, -0.38],  # F
    [ 2.23, -5.36,  0.30, -0.01,  1.21],  # G
    [ 2.41,  1.74,  1.11, -1.25,  1.80],  # H
    [-4.44, -1.68, -1.03, -0.72,  0.84],  # I
    [ 2.84,  1.41, -3.14, -0.67,  2.45],  # K
    [-4.19, -1.03, -0.98, -0.37,  1.40],  # L
    [-2.49, -0.27, -0.41,  1.94,  0.23],  # M
    [ 3.22,  1.45,  0.84, -0.72,  0.89],  # N
    [-1.22,  0.88,  2.23,  0.00, -1.65],  # P
    [ 2.18,  0.53, -1.14,  1.62, -0.09],  # Q
    [ 2.88,  2.52, -3.44,  1.95, -0.15],  # R
    [ 1.96, -1.63,  0.57, -0.96,  0.43],  # S
    [ 0.92, -2.09, -0.00, -1.47,  0.84],  # T
    [-2.69, -2.53, -1.29, -0.11,  1.31],  # V
    [-4.75,  3.65,  0.85,  0.97, -0.80],  # W
    [-1.39,  2.32,  0.01, -0.15,  3.51],  # Y
], dtype=np.float32)


def extract_classic_features(token_id: int) -> np.ndarray:
    """
    Extract handcrafted features for a single token.
    
    Args:
        token_id: Token ID from the vocabulary
        
    Returns:
        Feature vector of shape (45,): [one_hot(20), blosum62(20), zscale(5)]
    """
    features = np.zeros(45, dtype=np.float32)
    
    # Handle special tokens (PAD, CLS, SEP, MASK) - return zero vector
    if token_id in SPECIAL_IDS or token_id == 0:
        return features
    
    # Map token ID to amino acid
    if token_id not in TOKEN_TO_AA:
        # Unknown token - return zero vector
        return features
    
    aa = TOKEN_TO_AA[token_id]
    if aa not in AA_TO_IDX:
        # Unknown amino acid - return zero vector
        return features
    
    aa_idx = AA_TO_IDX[aa]
    
    # One-hot encoding (positions 0-19)
    features[aa_idx] = 1.0
    
    # BLOSUM62 row (positions 20-39)
    features[20:40] = BLOSUM62[aa_idx]
    
    # Z-scale (positions 40-44)
    features[40:45] = ZSCALE[aa_idx]
    
    return features


def process_npz_file(npz_path: str) -> None:
    """
    Add classic_feats to an existing NPZ file.
    
    Args:
        npz_path: Path to the NPZ file to process
    """
    print(f"Processing: {npz_path}")
    
    # Load existing data
    data = np.load(npz_path)
    inputs = data['inputs']  # Shape: [N, max_len]
    labels = data['labels']  # Shape: [N]
    
    # Check if classic_feats already exists
    if 'classic_feats' in data:
        print(f"  classic_feats already exists in {npz_path}, skipping...")
        return
    
    N, max_len = inputs.shape
    print(f"  Shape: {N} samples x {max_len} tokens")
    
    # Initialize classic_feats array
    classic_feats = np.zeros((N, max_len, 45), dtype=np.float32)
    
    # Extract features for each token
    print(f"  Extracting features...")
    for i in range(N):
        for t in range(max_len):
            token_id = int(inputs[i, t])
            classic_feats[i, t] = extract_classic_features(token_id)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{N} samples")
    
    # Save updated NPZ file with all original keys plus classic_feats
    print(f"  Saving updated file...")
    np.savez_compressed(
        npz_path,
        inputs=inputs,
        labels=labels,
        classic_feats=classic_feats
    )
    
    print(f"  ✓ Added classic_feats with shape {classic_feats.shape}")


def process_task(task: str, data_dir: str = "./data") -> None:
    """
    Process all splits for a given task.
    
    Args:
        task: Task name ('hemo', 'sol', 'nf')
        data_dir: Root data directory
    """
    task_dir = os.path.join(data_dir, task)
    
    if not os.path.exists(task_dir):
        print(f"Error: Task directory not found: {task_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing task: {task}")
    print(f"{'='*60}")
    
    for split in ['train', 'val', 'test']:
        npz_path = os.path.join(task_dir, f"{split}.npz")
        
        if not os.path.exists(npz_path):
            print(f"Warning: {npz_path} not found, skipping...")
            continue
        
        process_npz_file(npz_path)
    
    print(f"\n✓ Completed processing for task '{task}'")


def main():
    parser = argparse.ArgumentParser(
        description='Add handcrafted features (one-hot + BLOSUM62 + Z-scale) to NPZ files'
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['hemo', 'sol', 'nf'],
        help='Task name'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Root data directory (default: ./data)'
    )
    
    args = parser.parse_args()
    
    process_task(args.task, args.data_dir)
    
    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
