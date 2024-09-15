import certifi
import ssl

context = ssl.create_default_context(cafile=certifi.where())

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import os
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

# Set up a local directory for caching models
cache_dir = os.path.expanduser("~/.cache/shap_e_models")
os.makedirs(cache_dir, exist_ok=True)

device = torch.device('mps')  # 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Load models with local caching
xm = load_model('transmitter', device=device, cache_dir=cache_dir)
model = load_model('text300M', device=device, cache_dir=cache_dir)
diffusion = diffusion_from_config(load_config('diffusion', cache_dir=cache_dir))

batch_size = 4
guidance_scale = 15.0
prompt = "a shark"

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
    device=device,  # Explicitly pass the device
)

render_mode = 'nerf'  # you can change this to 'stf'
size = 64  # this is the size of the renders; higher values take longer to render.

cameras = create_pan_cameras(size, device)

for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    display(gif_widget(images))

# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh

for i, latent in enumerate(latents):
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'example_mesh_{i}.ply', 'wb') as f:
        t.write_ply(f)
    with open(f'example_mesh_{i}.obj', 'w') as f:
        t.write_obj(f)