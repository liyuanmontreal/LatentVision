# viz_color_space.py
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.decomposition import PCA

###############################################################
# 1) 3D Embedding Cloud
###############################################################
def plot_3d_color_space(X_low):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        X_low[:,0], X_low[:,1], X_low[:,2],
        c=X_low, s=20
    )

    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title("3D Embedding Cloud")
   

    return fig


###############################################################
# 2) 2D Projection
###############################################################
def plot_2d_color_map(X_low):
    p = PCA(n_components=2)
    X2 = p.fit_transform(X_low)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(X2[:,0], X2[:,1], c=X_low, s=10)
    ax.set_title("2D Projection")
    ax.axis("off")

    return fig


###############################################################
# 3) RGB vs AI Embedding Distribution
###############################################################
def plot_rgb_cube_and_ai(X_low):
    r = np.linspace(0, 1, 8)
    R, G, B = np.meshgrid(r, r, r)
    rgb_cube = np.vstack([R.flatten(), G.flatten(), B.flatten()]).T

    fig = plt.figure(figsize=(12,5))

    # 人类 RGB 立方体
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(rgb_cube[:,0], rgb_cube[:,1], rgb_cube[:,2], c=rgb_cube, s=15)
    ax1.set_title("Human RGB Cube")

    # AI Embedding Distribution
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X_low[:,0], X_low[:,1], X_low[:,2], c=X_low, s=15)
    ax2.set_title("Embedding Distribution")   
       

    return fig# viz_color_space.py
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.decomposition import PCA


def plot_3d_color_space(X_low):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_low[:,0], X_low[:,1], X_low[:,2], c=X_low, s=10)
    ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
    ax.set_title("3D Embedding Cloud")
    return fig


def plot_2d_color_map(X_low):
    p = PCA(n_components=2)
    X2 = p.fit_transform(X_low)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(X2[:,0], X2[:,1], c=X_low, s=10)
    ax.set_title("2D Projection")
    ax.axis("off")
    return fig


def plot_rgb_cube_and_ai(X_low):
    r = np.linspace(0, 1, 6)
    R, G, B = np.meshgrid(r, r, r)
    rgb_cube = np.vstack([R.flatten(), G.flatten(), B.flatten()]).T

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(rgb_cube[:,0], rgb_cube[:,1], rgb_cube[:,2], c=rgb_cube, s=10)
    ax1.set_title("Human RGB Cube")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X_low[:,0], X_low[:,1], X_low[:,2], c=X_low, s=10)
    ax2.set_title("Embedding Distribution")
    return fig


# === 新增可视化工具 ===
from sklearn.cluster import KMeans
import plotly.express as px


def plot_color_distance_heat(X_low):
    """色彩距离热力云"""
    # 计算距离中心点的距离
    center = np.mean(X_low, axis=0)
    dist = np.linalg.norm(X_low - center, axis=1)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(X_low[:,0], X_low[:,1], X_low[:,2], c=dist, cmap="inferno", s=30)
    fig.colorbar(p, ax=ax, shrink=0.6)
    ax.set_title("Embedding Distance Heatmap")
    
        
    return fig


def plot_clustered_color_space(X_low, n_clusters=6):
    """透明点云 + 聚类星团"""
    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(X_low)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        X_low[:,0], X_low[:,1], X_low[:,2],
        c=labels, cmap='tab10', s=40, alpha=0.45  # 透明点云
    )
    ax.set_title(f" K-Means Clusters ({n_clusters} groups)")
    return fig


def plot_plotly_3d(X_low):
    """Plotly 交互 3D 色云"""
    fig = px.scatter_3d(
        x=X_low[:,0], y=X_low[:,1], z=X_low[:,2],
        color=X_low[:,2],  # 用蓝色通道着色，也可以换 cluster
        opacity=0.7,
        title="Interactive Embedding Cloud (Plotly)"
    )
    fig.update_traces(marker=dict(size=5))
    return fig

