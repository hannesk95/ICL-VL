"""
Build a <anchor + 31 neighbours> table once and store it on disk.

Assumes you already have:
  • images/ ── your dataset     (recursive search)
  • index.faiss / meta.npy     (built earlier with DINOv2, etc.)

Run:
  python build_knn_table.py \
         --index embeddings/CRC100K_binary_dinov2_224.faiss \
         --meta  embeddings/CRC100K_binary_dinov2_224.faiss.meta.npy \
         --out   data/knn_31_similar.csv \
         --k     31
"""
import argparse, os, csv, faiss, numpy as np

def load_index(index_path, meta_path):
    index = faiss.read_index(index_path)
    ids   = np.load(meta_path, allow_pickle=True)
    return index, ids          # ids[i] is the absolute image path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--meta",  required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--k",     type=int, default=31)
    args = ap.parse_args()

    index, paths = load_index(args.index, args.meta)
    k = args.k

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["anchor"] + [f"nn{i+1}" for i in range(k)])

        # batch-wise is faster; here is the naive loop for clarity
        for i, anchor_path in enumerate(paths):
            D, I = index.search(index.reconstruct(i)[None, :], k+1)  # +1 b/c itself
            nbrs = [paths[j] for j in I[0] if j != i][:k]
            wr.writerow([anchor_path] + nbrs)

if __name__ == "__main__":
    main()