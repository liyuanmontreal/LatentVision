# -*- coding: utf-8 -*-
import os, json, argparse, numpy as np
from PIL import Image
from expandcolor.alien_color import encode_image_to_latents, project_to_display
from expandcolor.metrics import evaluate_all

def main():
    ap = argparse.ArgumentParser(description="Evaluate projection quality for ExpandColor")
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--X_high", type=str, default=None)
    ap.add_argument("--X_low", type=str, default=None)
    ap.add_argument("--latent-dim", type=int, default=384)
    ap.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--method", type=str, choices=["pca","umap"], default="pca")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--triplets", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="outputs/eval_metrics.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        img = img.resize((224, 224))  # ✅ 强制适配 ViT 输入尺寸
        latents, _ = encode_image_to_latents(img, model_name=args.model_name, pretrained=True,
                                             latent_dim=args.latent_dim, device=args.device)
        X_high = latents
        X_low = project_to_display(X_high, method=args.method, seed=args.seed)
    else:
        if args.X_high is None or args.X_low is None:
            ap.error("Provide --image or both --X_high and --X_low")
        X_high = np.load(args.X_high)
        X_low  = np.load(args.X_low)
        assert X_high.shape[0]==X_low.shape[0], "Mismatched N"

    metrics = evaluate_all(X_high, X_low, k=args.k, triplets=args.triplets, seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({k:(None if isinstance(v,float) and (np.isnan(v) if isinstance(v,float) else False) else v)
                   for k,v in metrics.items()}, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
