# ExpandColor — Vision Transformer Gamut Expansion (非人类色彩空间)

> RGB → ViT latent (N-d) → 聚类/投影 → 人眼可视化（“翻译后的外星色”）

## Features
- ViT encoder via `timm` (or `torchvision` fallback) → patch embeddings
- KMeans to invent **AI color families**
- PCA/UMAP to project high-dim latent → display RGB
- Recolored image + invented color palette with auto-naming
- CLI + Streamlit app
- ✅ **Projection quality metrics**（Stress/Trustworthiness/Continuity/LCMC/Triplet）

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/alien_palette.py --image flower.png --clusters 16 --latent-dim 384 --outdir outputs
```
you can change flower.png to other image file

## Evaluation（评估投影质量）
从图像直接评估：
```bash
python scripts/evaluate_projection.py --image flower.png --method pca --k 15 --triplets 50000 --out outputs/eval_metrics.json
```
或对已有高/低维表示评估：
```bash
python scripts/evaluate_projection.py --X_high X_high.npy --X_low X_low.npy --k 15
```

输出 JSON 指标：`stress`、`trustworthiness@k`、`continuity@k`、`lcmc@k`、`triplet_preservation`。


