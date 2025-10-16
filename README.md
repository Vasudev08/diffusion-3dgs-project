# Diffusion-Enchanced 3D Guassian Splatting

## Setup

```bash
git clone --recursive https://github.com/Vasudev08/diffusion-3dgs-project.git
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e stable-virtual-camera
pip install -e gsplat
```

### With uv

```bash
uv venv
source .venv/bin/activate
uv pip install -e stable-virtual-camera
uv pip install -e gsplat --no-build-isolation
```

### For development

Install prek (https://github.com/j178/prek), then install the git hook alongside the hook environments with

```bash
prek install --install-hooks
```
