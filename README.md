# ai-research
Personal repo containing AI research and exploration

## Setup
**Note**: It is assumed that this is cloned into an environment that already has jupyter and pytorch setup; e.g., a "deeplearning-platform-release" GCE instance w/ jupyter setup on startup.

### Local dev
This project uses [uv](https://docs.astral.ai/uv/) as its Python package and project manager (a fast replacement for `pip`/`venv`/`pip-tools`). Dependencies are declared in [pyproject.toml](pyproject.toml) and locked in [uv.lock](uv.lock).

1. Install uv (macOS):
```shell
brew install uv
```
Or via the official installer:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Sync the project's virtualenv (creates `.venv/` and installs locked deps):
```shell
uv sync
```
3. Run commands inside the project env:
```shell
uv run python main.py
# or activate the venv directly:
source .venv/bin/activate
```
4. Add / remove dependencies (updates `pyproject.toml` and `uv.lock`):
```shell
uv add <package>
uv remove <package>
```

### Notebooks via VSCode + jupytext
Rather than committing `.ipynb` files (noisy diffs from cell outputs / metadata), this project uses python interactive cells (# %%). We can optionally use [jupytext](https://jupytext.readthedocs.io/) to pair notebooks with plain-text representations (e.g. `.py` percent-format or `.md`) that are friendly to git review.

1. Make sure your `.venv` is selected as the Python interpreter in VSCode (`Cmd+Shift+P` → "Python: Select Interpreter" → `.venv`).
2. Install the **Jupyter** extension (`ms-toolsai.jupyter`) — it natively opens jupytext-paired files as interactive notebooks.
3. Open a paired text file (e.g. `llm.py` with a `# %%` percent-format header) and VSCode will render it as a notebook with runnable cells against the project kernel.
4. To pair a new notebook with a text representation:
```shell
uv run jupytext --set-formats ipynb,py:percent <notebook>.ipynb
uv run jupytext --sync <notebook>.ipynb
```
Commit the `.py` (or `.md`) file; the `.ipynb` can be gitignored or regenerated on demand with `jupytext --sync`.

### Connecting to a VM
If interacting with this on a VM, users will likely need access to their Huggingface access tokens, git credentials and more.

1. Connect
```shell
gcloud compute ssh --project playground-dev-6ae7 --zone us-central1-a ai-dev-vm -- -L 8080:localhost:8080
```
2. The VM I actually use has a template systemd service setup via this startup script:
```
#!/bin/bash
set -euxo pipefail
LOGFILE="/var/log/jupyter-startup.log"
exec > >(tee -a $${LOGFILE} ) 2>&1

# Basic packages
apt-get update
apt-get install -y python3-venv python3-pip

# Create a shared virtualenv accessible by OS Login users
VENV_DIR="/opt/jupyterlab-venv"
python3 -m venv "$${VENV_DIR}"

# Make sure all users can access it
chmod -R a+rwx "$${VENV_DIR}"

# Activate venv and install JupyterLab
source "$${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install jupyterlab

# Create a systemd service for JupyterLab (runs as ubuntu user)
cat >/etc/systemd/system/jupyter@.service <<'EOF'
[Unit]
Description=JupyterLab for %i
After=network.target

[Service]
Type=simple
User=%i
WorkingDirectory=/home/%i
Environment="VENV_DIR=/opt/jupyterlab-venv"
# leave token ON (default) for safety; binding to localhost anyway
ExecStart=/opt/jupyterlab-venv/bin/jupyter lab --ip=127.0.0.1 --port=8080 --no-browser
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
```
After connecting, in the ssh terminal perform the following steps to startup the service for your oslogin user:
```shell
sudo systemctl enable jupyter@$(systemd-escape <your-linux-username>)

sudo systemctl start jupyter@$(systemd-escape <your-linux-username>)

sudo systemctl status jupyter@*
```

**Now, on localhost:8080 you should be able to perform the following in a JPL terminal:**

1. Expose your tokens on the session
```shell
export GIT_USERNAME='${GIT_USERNAME}'
export GIT_TOKEN='${GIT_TOKEN}'
export HUGGINGFACE_TOKEN='${HUGGINGFACE_TOKEN}'
```
2. Use HTTPS to interact with git
```shell
git clone https://$GIT_USERNAME:$GIT_TOKEN@github.com/bradlet/ai-research.git
```
3. In a notebook cell, you can run `%pip install -r requirements.txt` to get access to any required Python modules in the current session.

