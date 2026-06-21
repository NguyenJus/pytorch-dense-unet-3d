# Install the package in a fresh virtual environment (Windows / PowerShell).
# Dependencies are declared in pyproject.toml; requirements.txt has been removed.
# For the CPU-only PyTorch wheel use the --extra-index-url flag below.

python -m venv venv-dense_unet_3d
venv-dense_unet_3d\Scripts\activate.ps1
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
