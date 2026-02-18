# nanomup

A nano-scale implementation to reproduce experiments from **Feature Learning in Infinite-Width Neural Networks** (μP / MAML), based on [TP4](https://github.com/edwardjhu/TP4).

---

## Configuring the environment

This project uses **[uv](https://docs.astral.sh/uv/)** for dependency and environment management. All dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

### Prerequisites

- **Python 3.12+**
- **uv** (recommended: install via the official installer)

### Install uv

If you don’t have uv yet:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Create and sync the environment

From the **project root** (the `nanomup` directory that contains `pyproject.toml`):

```bash
cd nanomup
uv sync
```

This will:

- Create a virtual environment (e.g. `.venv` in the project root) if it doesn’t exist
- Install the exact versions from `uv.lock` (Python 3.12+, PyTorch, torchvision, matplotlib)

### Run commands in the environment

Use `uv run` so that commands use the project’s environment automatically:

```bash
# Run a script
uv run python script.py

# Run notebooks (from project root or notebooks/)
uv run jupyter notebook
# or
uv run jupyter lab
```

To activate the virtual environment in your shell instead:

```bash
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate     # Windows
```

Then run `python`, `jupyter`, etc. as usual.

### Optional: pin Python version with uv

To use a specific Python version managed by uv:

```bash
uv python pin 3.12
uv sync
```
