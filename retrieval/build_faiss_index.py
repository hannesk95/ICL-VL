# retrieval/build_faiss_index.py
import numpy as np, faiss, os

EMB_FILE = "embeddings/CRC100K_binary_dinov2_224.npz"
INDEX_OUT = "embeddings/CRC100K_binary_dinov2_224.faiss"

data = np.load(EMB_FILE)
features = data["feats"].astype("float32")       # faiss wants float32
index = faiss.IndexFlatIP(features.shape[1])     # cosine via inner-product
faiss.normalize_L2(features)                     # cosine equivalently dot on ℓ2-normed
index.add(features)
faiss.write_index(index, INDEX_OUT)
np.save(INDEX_OUT + ".meta.npy", np.stack([data["paths"], data["labels"]], axis=1))