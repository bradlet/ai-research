# %% [markdown]
# AI Research

# %%
import torch

from lib import MemoryOffloader, SimpleWriter

# %%
assert torch.backends.mps.is_available(), "MPS backend is not available"
assert torch.backends.mps.is_built(), "PyTorch was not built with MPS support"
device = torch.device("mps")
print(f"Using device: {device}")

# %%
a = torch.randn(4, 4, device=device)
b = torch.randn(4, 4, device=device)
c = a @ b
print(f"a.device={a.device}, b.device={b.device}, c.device={c.device}")
print(c)

# %%
mol = MemoryOffloader()

x = "/Users/bradley/projects/ai-research/temp"
mol[x] = {
  "foo": "test",
  "bar": "test"
}
print(mol[x])
