#!/usr/bin/env bash
set -e

echo "Installing PyTorch (cu121)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installing basicsr (editable, no build isolation)..."
python -m pip install --no-build-isolation -e . --no-deps

echo "Verifying GPU + basicsr..."
python -c "import torch; print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
python -c "from basicsr.data import create_dataloader, create_dataset; print('basicsr import: OK')"

echo "Setup complete. Now run:"
echo "export PYTHONPATH=\$(pwd):\$PYTHONPATH"
