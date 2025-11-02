# app_space.py —— LatentVision Viewer
# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import io
import base64
import streamlit as st
import numpy as np
from PIL import Image

from latentvision.alien_color import encode_image_to_latents, project_to_display
from viz_space import (
    plot_3d_color_space, plot_2d_color_map, plot_rgb_cube_and_ai,
    plot_color_distance_heat, plot_clustered_color_space, plot_plotly_3d
)

# ---- Page config ----
st.set_page_config(page_title="LatentVision Explorer", layout="wide")
st.title("🔍 LatentVision Explorer")
st.caption("Explore and interpret high-dimensional vision embeddings by projection into human-interpretable geometric spaces.")

uploaded = st.file_uploader("Load image", type=["jpg","jpeg","png","webp"])

clusters = st.slider("Embedding groups", 4, 32, 16)
latent_dim = st.select_slider("Embedding dimension (ViT)", [192,384,768,1024], 384)
method = st.selectbox("Projection", ["pca", "umap"])
device = "cpu"

from PIL import Image

def prepare_image(img, size=224):
    # 等比例缩放
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # 创建黑色背景 + 居中粘贴
    new_img = Image.new("RGB", (size, size), (0, 0, 0))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


# ---- Process after upload ----
if uploaded is not None:
    with st.expander("📎 What is happening?"):
        st.markdown("""
            **Patch → Embedding → Projection**

            1. The image is sliced ​​into 16×16 patches. 
            2. Each patch enters the Vision Transformer, resulting in a high-dimensional embedding vector.            
            3. Projection is performed using PCA/UMAP onto 2D/3D spaces.
            4. revealing the model's "feature space structure."    
             
            Goal: To understand how the model organizes visual semantics.
            """)


    img = Image.open(uploaded).convert("RGB")
    #img = img.resize((224, 224))  # ViT input constraint
    img_display = img.copy()  # 用来展示原图
    img = prepare_image(img)  # 用等比例缩放的图片用于模型
    st.image(img_display, caption="输入图像 / Input image", width=400)

    latents, _ = encode_image_to_latents(img, latent_dim=latent_dim, device=device)
    X_low = project_to_display(latents, method=method)

    st.write(f"✅ Embedding shape: {X_low.shape}, range: [{X_low.min():.3f}, {X_low.max():.3f}]")

    # ------ Export Features ------
    st.subheader("📁 Export")

    # Export embeddings as numpy file
    emb_buf = io.BytesIO()
    np.save(emb_buf, X_low)
    st.download_button(
        "⬇️ Download Embeddings (.npy)",
        data=emb_buf.getvalue(),
        file_name="latent_embeddings.npy"
    )

    # Export raw latents
    lat_buf = io.BytesIO()
    np.save(lat_buf, latents)
    st.download_button(
        "⬇️ Download Raw Latents (.npy)",
        data=lat_buf.getvalue(),
        file_name="vit_latents.npy"
    )

    # Export current figure button (inject when a figure exists)
    def save_fig_btn(fig, name="latentvision_plot.png"):
        buf = io.BytesIO()

        # Matplotlib figure?
        if hasattr(fig, "savefig"):
            fig.savefig(buf, format="png", dpi=200)
            data = buf.getvalue()
            st.download_button(f"🖼 Download {name}", data=data, file_name=name)
            return
        
        # Plotly figure
        try:
            import plotly.io as pio
            img_bytes = pio.to_image(fig, format="png", width=900, height=700)
            st.download_button(f"🖼 Download {name}", data=img_bytes, file_name=name)
            return
        except Exception as e:
            st.warning(f"⚠️ Unable to export this figure type ({type(fig)}). Details: {e}")

    # ---- Tabs ----
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "3D Embedding Cloud",
        "2D Projection",
        "RGB vs Embedding Distribution",
        "Embedding Distance Heatmap",
        "Cluster Groups",
        "Interactive Embedding Cloud (Plotly)"
    ])

    with tab1:
        st.subheader("High-dimensional visual embeddings projected to 3D")
        fig = plot_3d_color_space(X_low)
        st.pyplot(fig, use_container_width=False)
        save_fig_btn(fig, "3d_embedding.png")


    with tab2:
        st.subheader("2D Projection of embedding manifold")
        fig = plot_2d_color_map(X_low)
        st.pyplot(fig, use_container_width=False)
        save_fig_btn(fig, "2d_embedding.png")


    with tab3:
        st.subheader("Human RGB space vs Model embedding distribution")
        fig = plot_rgb_cube_and_ai(X_low)
        st.pyplot(fig, use_container_width=False)
        save_fig_btn(fig, "RGB_vs_Model_embedding.png")


    with tab4:
        st.subheader("Embedding distance heatmap")
        fig = plot_color_distance_heat(X_low)
        st.pyplot(fig, use_container_width=False)
        save_fig_btn(fig, "embedding_distance_heatmap.png")


    with tab5:
        st.subheader("K-Means cluster groups in embedding space")
        fig = plot_clustered_color_space(X_low, n_clusters=clusters)
        st.pyplot(fig, use_container_width=False)
        save_fig_btn(fig, "K-Means_cluster_groups.png")


    with tab6:
        st.subheader("Interactive 3D embedding cloud (Plotly)")
        fig = plot_plotly_3d(X_low)
        st.plotly_chart(fig, use_container_width=False)
        save_fig_btn(fig, "Interactive_3D_embedding.png")

    with st.expander(" About LatentVision / README"):
        st.markdown("""
            ### LatentVision: Visual Embedding Exploration Toolkit

            This tool visualizes high-dimensional vision embeddings  
            and projects them into interpretable geometric spaces.

            **Key ideas**
            - Vision Transformer embeddings
            - PCA / UMAP projection
            - Embedding topology & clusters
            - Model interpretability

            **Tech stack**
            PyTorch • Timm • NumPy • sklearn • UMAP • Streamlit • Plotly

            """)

else:
    st.info("📎Upload an image to begin")
