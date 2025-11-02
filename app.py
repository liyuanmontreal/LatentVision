# -*- coding: utf-8 -*-
import os, json
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from latentvision.alien_color import pipeline

st.set_page_config(page_title="LatentVision", layout="wide")
st.title("Latent Vision Explorer （Vision Embedding Manifold）")
st.caption("RGB → ViT latent → cluster → project → translate to human display")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])
col1, col2 = st.columns([1,1])

with st.sidebar:
    st.markdown("### Settings")
    clusters = st.slider("Clusters", 4, 32, 16, 1)
    latent_dim = st.select_slider("Latent dim", options=[192,384,768,1024], value=384)
    model_name = st.text_input("ViT model name", "vit_base_patch16_224")
    use_umap = st.checkbox("Use UMAP", value=False)
    seed = st.number_input("Seed", value=42, step=1)
    device = st.selectbox("Device", options=["cpu","cuda"], index=0)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    col1.image(img, caption="Input", use_column_width=True)
    tmp = "tmp_input.png"; img.save(tmp)
    if st.button("Generate"):
        with st.spinner("Encoding → clustering → projection → recolor…"):
            res = pipeline(tmp, "outputs", k=clusters, latent_dim=latent_dim,
                           model_name=model_name, use_umap=use_umap, seed=seed, device=device)
        col2.image(Image.open(res["image"]), caption="Translated image", use_column_width=True)
        st.image(Image.open(res["palette"]), caption="Invented palette")
        with open(res["json"], "r", encoding="utf-8") as f:
            st.json(json.load(f))
else:
    st.info("Upload an image to start.")
