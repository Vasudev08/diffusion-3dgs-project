# Diffusion-Enchanced 3D Guassian Splatting

## Setup

```bash
git clone --recursive https://github.com/Vasudev08/diffusion-3dgs-project.git
```

If the repository is already cloned but the submodules are not initialized, run:

```bash
git submodule update --init --recursive
git submodule update --remote --recursive
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate
```

Change the numpy version requirement in the `pyproject.toml` of stable-virtual-camera from `"numpy==1.24.4"` to `"numpy>=1.24.4"`.

```bash
pip install -e stable-virtual-camera
pip install -e gsplat
```

### With uv

```bash
uv venv
source .venv/bin/activate
```

Change the numpy version requirement in the `pyproject.toml` of stable-virtual-camera from `"numpy==1.24.4"` to `"numpy>=1.24.4"`.

```bash
uv pip install -e stable-virtual-camera
uv pip install -e gsplat --no-build-isolation
```

### For development

Install prek (https://github.com/j178/prek), then install the git hook alongside the hook environments with

```bash
prek install --install-hooks
```
