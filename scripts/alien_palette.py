# -*- coding: utf-8 -*-
import argparse, os, json
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from latentvision.alien_color import pipeline

def main():
    p = argparse.ArgumentParser(description="ExpandColor — ViT Gamut Expansion")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--clusters", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=384)
    p.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    p.add_argument("--umap", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    res = pipeline(a.image, a.outdir, k=a.clusters, latent_dim=a.latent_dim,
                   model_name=a.model_name, use_umap=a.umap, seed=a.seed, device=a.device)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
