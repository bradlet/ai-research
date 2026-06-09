# %% [markdown]
# # Stable Diffusion 3.5 Large on MPS
#
# Take a hardcoded prompt, generate an image with SD 3.5 Large via 🤗 diffusers
# on Apple Silicon (MPS), and save it to `out/{random}.png`.

# %% [markdown]
# ## Hugging Face web-UI prerequisites (one-time)
#
# 1. Create an account at https://huggingface.co.
# 2. Open the model page: https://huggingface.co/stabilityai/stable-diffusion-3.5-large
#    and click **"Agree and access repository"** to accept Stability's Community License.
#    The model is gated — without this step `from_pretrained` will raise `GatedRepoError`.
# 3. Generate a **Read** access token at https://huggingface.co/settings/tokens.
# 4. Copy `.env.example` to `.env` at the repo root and paste your token:
#    ```shell
#    cp .env.example .env
#    # then edit .env so HUGGINGFACE_TOKEN=hf_...
#    ```
#    `.env` is gitignored. The script loads it via `python-dotenv` below.

# %% [markdown]
# ## Local prerequisites
#
# - One-time: `uv add diffusers accelerate python-dotenv`
# - Disk: model weights are ~16GB; the first run downloads them into the
#   Hugging Face cache (`~/.cache/huggingface`).

# %%
import os
import secrets
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from dotenv import load_dotenv
from huggingface_hub import login

# https://huggingface.co/stabilityai/stable-diffusion-3.5-large
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
PROMPT = "A giant monster which looks like a flying ant with scorpion arms and a scorpion tail. The monster's face is dominating by menacing pincers and spines."
OUT_DIR = Path(__file__).parent / "out"
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5

# %%
assert torch.backends.mps.is_available(), "MPS backend is not available"
assert torch.backends.mps.is_built(), "PyTorch was not built with MPS support"
device = torch.device("mps")
print(f"Using device: {device}")

# %%
load_dotenv()
login(token=os.environ["HUGGINGFACE_TOKEN"])

# %% [markdown]
# ## Loading the pipeline
#
# - `bfloat16` is the dtype the SD 3.5 model card recommends.
# - `enable_attention_slicing()` is the HF MPS-doc recommendation for any Mac
#   without 64GB+ unified memory; SD 3.5 Large is big enough that this matters
#   even on higher-end M-series.
# - If you hit OOM, swap `MODEL_ID` to `stabilityai/stable-diffusion-3.5-medium`
#   (same pipeline class, ~2.5B params instead of ~8B).

# %%
pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

# %% [markdown]
# ## Generate
#
# Expect multi-minute latency on the first run (compile + warmup); subsequent
# runs are faster.

# %%
image = pipe(
  PROMPT,
  num_inference_steps=NUM_INFERENCE_STEPS,
  guidance_scale=GUIDANCE_SCALE,
).images[0]

# %%
OUT_DIR.mkdir(exist_ok=True)
out_path = OUT_DIR / f"{secrets.token_hex(8)}.png"
image.save(out_path)
print(f"Saved: {out_path}")
