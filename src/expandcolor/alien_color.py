# -*- coding: utf-8 -*-
from typing import Tuple, Dict, Any, Optional
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import os, math, json, numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
import torch, torch.nn.functional as F
from .model import ViTEncoder

def load_image(path: str, max_side: int = 768) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))  # ✅ 强制适配 ViT 输入尺寸
    w,h = img.size
    s = min(1.0, max_side/ max(w,h))
    if s < 1.0:
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return img

def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32)/255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

def auto_name_color(rgb: np.ndarray) -> str:
    r,g,b = rgb.tolist()
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    idx = int(np.argmax([r,g,b]))
    vibe = ["Dream","Void","Fractal","Solar","Memory","Nebula","Echo","Selenia"]
    base = ["Red","Green","Blue"]
    return f"{vibe[int((lum*7))%len(vibe)]} {base[idx]}"

def make_palette_image(colors: np.ndarray, names=None, swatch: int=80, cols: int=8) -> Image.Image:
    k = colors.shape[0]
    rows = (k + cols - 1)//cols
    img = Image.new("RGB", (cols*swatch, rows*swatch + 24), (255,255,255))
    d = ImageDraw.Draw(img)
    for i, c in enumerate(colors):
        r = int(c[0]*255); g = int(c[1]*255); b = int(c[2]*255)
        x = (i % cols)*swatch; y = (i//cols)*swatch
        d.rectangle([x,y,x+swatch,y+swatch], fill=(r,g,b))
    if names:
        text = " | ".join(names)
        d.text((8, rows*swatch+4), text[:120], fill=(0,0,0))
    return img

def encode_image_to_latents(img: Image.Image, model_name="vit_base_patch16_224",
                            pretrained=True, latent_dim=384, device="cpu"):
    enc = ViTEncoder(model_name=model_name, pretrained=pretrained, out_dim=latent_dim).to(device)
    x = to_tensor(img).to(device)
    _,_,H,W = x.shape
    Hc = (H//16)*16; Wc = (W//16)*16
    x = F.interpolate(x, size=(Hc,Wc), mode="bilinear", align_corners=False)
    feats, (Hp, Wp) = enc(x, return_hw=True)
    return feats[0].detach().cpu().numpy(), (Hp, Wp)

def cluster_colors(latents: np.ndarray, k: int=16, seed: int=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(latents)
    return km.cluster_centers_, labels

def project_to_display(vectors: np.ndarray, method="pca", seed: int=42) -> np.ndarray:
    if method=="umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=3, random_state=seed, metric="cosine")
        y = reducer.fit_transform(vectors)
    else:
        reducer = PCA(n_components=3, random_state=seed)
        y = reducer.fit_transform(vectors)
    y = (y - y.min(0)) / (y.max(0) - y.min(0) + 1e-8)
    return y

def reconstruct_image(img: Image.Image, labels: np.ndarray, hw, centroids_rgb: np.ndarray) -> Image.Image:
    Hp, Wp = hw
    patch_h = img.size[1] // Hp
    patch_w = img.size[0] // Wp
    canvas = Image.new("RGB", (Wp*patch_w, Hp*patch_h))
    idx = 0
    for i in range(Hp):
        for j in range(Wp):
            c = tuple((centroids_rgb[labels[idx]]*255).astype(np.uint8).tolist())
            tile = Image.new("RGB", (patch_w, patch_h), c)
            canvas.paste(tile, (j*patch_w, i*patch_h))
            idx += 1
    return canvas

def pipeline(image_path: str, outdir: str, k: int=16, latent_dim: int=384,
             model_name: str="vit_base_patch16_224", use_umap: bool=False, seed: int=42, device: str="cpu"):
    os.makedirs(outdir, exist_ok=True)
    img = load_image(image_path)
    latents, hw = encode_image_to_latents(img, model_name=model_name, pretrained=True,
                                          latent_dim=latent_dim, device=device)
    centroids, labels = cluster_colors(latents, k=k, seed=seed)
    rgb = project_to_display(centroids, method="umap" if use_umap else "pca", seed=seed)
    names = [auto_name_color(c) for c in rgb]

    pal_img = make_palette_image(rgb, names=names)
    pal_path = f"{outdir}/translated_palette.png"; pal_img.save(pal_path)
    rec = reconstruct_image(img, labels, hw, rgb)
    rec_path = f"{outdir}/translated_image.png"; rec.save(rec_path)

    data = {"colors":[{"name":n,"rgb":c.tolist()} for n,c in zip(names, rgb)],
            "k":k,"latent_dim":latent_dim,"model_name":model_name,"use_umap":use_umap,"seed":seed}
    json_path = f"{outdir}/colors.json"
    with open(json_path,"w",encoding="utf-8") as f: import json; json.dump(data,f,ensure_ascii=False,indent=2)
    return {"palette": pal_path, "image": rec_path, "json": json_path, "latents_shape": list(latents.shape)}
