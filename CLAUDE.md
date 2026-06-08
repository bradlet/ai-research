# CLAUDE.md

Personal AI-research playground. Each top-level `.py` is a self-contained interactive script for one experiment.

## Conventions
- **Interactive scripts only** — no `.ipynb` committed. Use jupytext cells: `# %%` for code, `# %% [markdown]` for prose. Match the style of `main.py` (2-space indents, plain imports, no docstrings, comments only when a *why* is non-obvious).
- **Device** — target Apple Silicon MPS. Assert availability up front the way `main.py` does. Default dtypes: `bfloat16` for diffusion, `float16`/`float32` for smaller stuff.
- **Package manager** — `uv` only. Add deps with `uv add <pkg>` (never edit `pyproject.toml` by hand for additions). Run things via `uv run python <script>.py`.
- **Secrets** — load from gitignored `.env` via `python-dotenv`. Call `load_dotenv()` before `os.environ[...]`. New keys go in `.env.example` so the template stays in git.
- **Outputs** — write generated artifacts (images, weights, dumps) under `out/` (gitignored). Use `secrets.token_hex(8)` for random filenames; `Path(__file__).parent / "out"` for the dir.
- **Shared utilities** — `lib.py` holds reusable bits (currently `MemoryOffloader` for JSON-on-disk dict). Prefer extending `lib.py` over duplicating helpers across scripts.

## Things to know before suggesting code
- Gated HF models (e.g. SD 3.5) need the license accepted in the web UI before `from_pretrained` works — call this out in markdown cells, not just code.
- MPS is memory-sensitive: default to `pipe.enable_attention_slicing()` for diffusion pipelines unless the user says they have 64GB+ unified memory.
- README has VM/GCE setup details for running this remotely; check there before suggesting infra changes.
