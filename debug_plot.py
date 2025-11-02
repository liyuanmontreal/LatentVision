import streamlit as st
import numpy as np
from viz_space import plot_3d_color_space

# Fake data to test plot
X_low = np.random.rand(500, 3)

fig = plot_3d_color_space(X_low)

st.write("Expect colorful 3D point cloud below:")
st.pyplot(fig)
