# Install the package in a fresh virtual environment (Windows / PowerShell).
# Dependencies are declared in pyproject.toml; requirements.txt has been removed.
# Install PyTorch from its CPU-only index before the remaining dependencies.

python -m venv venv-dense_unet_3d
venv-dense_unet_3d\Scripts\activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
