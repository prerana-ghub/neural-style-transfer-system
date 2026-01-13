import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tempfile
import time
import torch

from nst import run_style_transfer_adam, run_fast_style_transfer, run_adain_pixel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import lpips
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='alex').to(device)

st.markdown("<h1 style='text-align: center; color: #4B0082;'>Neural Style Transfer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #FF69B4;'>Transform your images with art!</h3>", unsafe_allow_html=True)

# --- Model selection ---
model_choices = st.multiselect("Choose Style Transfer Models to Compare",
                               ["Classic NST (VGG)", "Fast Style Transfer", "AdaIN"],
                               default=["Classic NST (VGG)"])

content_file = st.file_uploader("Upload content image", type=["jpg","jpeg","png"])
style_file = st.file_uploader("Upload style image", type=["jpg","jpeg","png"])
size = st.slider("Image size", 128, 512, 256, step=64)
steps = st.slider("Steps", 10, 200, 50, step=10)
style_weight = st.slider("Style weight", 1e5, 5e6, 1e6)
content_weight = st.slider("Content weight", 0.1, 5.0, 1.0)

if content_file and style_file and st.button("Run Comparison"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as ctmp:
        ctmp.write(content_file.read())
        content_path = ctmp.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as stmp:
        stmp.write(style_file.read())
        style_path = stmp.name

    content_img = np.array(Image.open(content_path).resize((size, size))).astype(np.float32) / 255.0
    if content_img.ndim == 3 and content_img.shape[2] == 4:
        content_img = content_img[..., :3]

    results = {}
    for model in model_choices:
        start = time.time()
        if model == "Classic NST (VGG)":
            output, s_hist, c_hist, t_hist = run_style_transfer_adam(content_path, style_path, size=size,
                                                                     steps=steps, style_weight=style_weight,
                                                                     content_weight=content_weight)
            img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        elif model == "Fast Style Transfer":
            output, s_hist, c_hist, t_hist = run_fast_style_transfer(content_path, style_path, size=size,
                                                                     style_weight=style_weight,
                                                                     content_weight=content_weight)
            img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        elif model == "AdaIN":
            img, s_hist, c_hist, t_hist = run_adain_pixel(content_path, style_path, size=size)

        img = np.clip(img, 0, 1)
        elapsed = time.time() - start

        # Metrics
        ssim_score = ssim(content_img, img, channel_axis=2, data_range=1.0)
        psnr_score = psnr(content_img, img, data_range=1.0)

        # LPIPS
        c_tensor = torch.tensor(content_img).permute(2,0,1).unsqueeze(0).to(device)
        i_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device)
        lpips_score = lpips_model(c_tensor, i_tensor).item()

        results[model] = {"image": img, "time": elapsed, "ssim": ssim_score,
                          "psnr": psnr_score, "lpips": lpips_score,
                          "losses": (s_hist, c_hist, t_hist)}

    # --- Display side by side ---
    cols = st.columns(len(results))
    for i, (model, res) in enumerate(results.items()):
        with cols[i]:
            st.image(res["image"], caption=f"{model} Output", use_container_width=True)
            st.write(f"⏱️ {res['time']:.2f}s")
            st.write(f"SSIM: {res['ssim']:.4f}")
            st.write(f"PSNR: {res['psnr']:.2f} dB")
            st.write(f"LPIPS: {res['lpips']:.4f}")

    # --- Loss curves comparison ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, res in results.items():
        total_loss = res["losses"][2]
        if total_loss and len(total_loss) > 0:
            ax.plot(total_loss, label=f"{model} Total Loss")
        else:
            ax.plot([0.0] * 10, label=f"{model} (No Iterative Loss)")
    ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves Comparison")
    st.pyplot(fig)