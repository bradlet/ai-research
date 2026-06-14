# %% [markdown]
# # Stable Diffusion 3.5 Medium on MPS
#
# Take a list of prompts, generate an image for each with SD 3.5 Medium via
# 🤗 diffusers on Apple Silicon (MPS), and save them to `out/{random}.png`.

# %% [markdown]
# ## Hugging Face web-UI prerequisites (one-time)
#
# 1. Create an account at https://huggingface.co.
# 2. Open the model page: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
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
# - Disk: model weights are ~5GB; the first run downloads them into the
#   Hugging Face cache (`~/.cache/huggingface`).

# %%
import os
import secrets
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from dotenv import load_dotenv
from huggingface_hub import login

# https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
PROMPTS = [
  "Epic dark-fantasy key-art banner, ultra-wide cinematic composition. Two colossal champions face off across a storm-torn battlefield at the moment before impact, mirrored on the left and right edges. Left: a radiant god of light, armored in gold and dawn-fire, immense and serene, a halo of solar flame behind him. Right: a towering avatar of decay — a massive winged scorpion-like entity with a pincer-dominated face, black-violet carapace, dripping motes of entropic ash. Below, rivers of tiny kneeling mortal followers stream toward each champion, glowing gold on one side, sickly green on the other. The sky is split by a luminous thread-like loom of fate; floating runic glyphs crack the world's rules. Mythic and monumental, dramatic rim lighting, volumetric god-rays piercing storm clouds, painterly concept art, 8k. Palette: molten gold and amber vs decay-violet and entropy-green over storm-grey. Avoid: text, logos, UI, card frames, watermark, modern objects, cartoonish, flat.",
]

OUT_DIR = Path(__file__).parent / "out"
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 4.5

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
#   without 64GB+ unified memory.
# - Medium is ~2.5B params (vs ~8B for Large), so it loads and runs noticeably
#   lighter. If you have the headroom and want higher fidelity, swap `MODEL_ID`
#   to `stabilityai/stable-diffusion-3.5-large`.

# %%
pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

# %% [markdown]
# ## Generate
#
# Loop one prompt at a time rather than batching the whole list — MPS unified
# memory is the bottleneck, and a single image per forward pass keeps the
# footprint flat. Expect multi-minute latency on the first run (compile +
# warmup); subsequent prompts are faster.

# %%
OUT_DIR.mkdir(exist_ok=True)
for i, prompt in enumerate(PROMPTS):
  print(f"[{i + 1}/{len(PROMPTS)}] generating: {prompt[:60]}...")
  image = pipe(
    prompt,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
  ).images[0]
  out_path = OUT_DIR / f"{secrets.token_hex(8)}.png"
  image.save(out_path)
  print(f"  saved: {out_path}")
