try:
    import faiss                         # fast path
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    print("[WARN] faiss not found – using brute-force cosine search (slower)")

import numpy as np, timm, torch, torchvision.transforms as T
from PIL import Image
from pathlib import Path

class DinoRetriever:
    def __init__(self, index_path, meta_path, model_name="vit_small_patch14_dinov2",
                 img_size=224, device=None):
        from sklearn.preprocessing import normalize    # always available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load metadata (path, label) and features
        meta_npz = np.load(index_path.replace(".faiss", ".npz"))
        self.meta = np.stack([meta_npz["paths"], meta_npz["labels"]], axis=1)
        feats     = meta_npz["feats"].astype("float32")
        self.feats = normalize(feats)            # L2-normalise for cosine

        if _HAS_FAISS:
            import faiss
            self.index = faiss.IndexFlatIP(feats.shape[1])
            self.index.add(self.feats)

        self.encoder = timm.create_model(
            model_name, pretrained=True, img_size=img_size,
            dynamic_img_size=True, num_classes=0
        ).to(self.device).eval()

        self.tf = T.Compose([
            T.Resize(img_size), T.CenterCrop(img_size),
            T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)
        ])

    @torch.inference_mode()
    def _embed(self, img_pil):
        x = self.tf(img_pil).unsqueeze(0).to(self.device)
        v = self.encoder(x).cpu().numpy()
        v /= np.linalg.norm(v, axis=1, keepdims=True)   # L2-norm
        return v

    def query(self, img_pil, k=5, label_list=None, overquery_factor=5):
        q = self._embed(img_pil)

        if _HAS_FAISS:
            D, I = self.index.search(q, k * overquery_factor)
            idxs = I[0]
        else:                                           # brute-force
            sims = (self.feats @ q.T).ravel()
            idxs = np.argsort(-sims)[: k * overquery_factor]

        neighbours = []
        per_label_goal = (k + len(label_list) - 1)//len(label_list) if label_list else k

        for i in idxs:
            path, lbl = self.meta[i]
            if label_list is None or lbl in label_list:
                neighbours.append((Path(path), lbl))
                if len(neighbours) == k:
                    break
        return neighbours