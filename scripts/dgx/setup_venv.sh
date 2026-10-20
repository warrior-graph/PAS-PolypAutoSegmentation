#!/usr/bin/env bash
# Fallback when Docker is not available on the DGX Spark: a venv with PyTorch CUDA wheels
# for aarch64. Prefer scripts/dgx/run.sh (NGC container) when you can use Docker.
#
#   scripts/dgx/setup_venv.sh            # creates .venv-dgx
#   source .venv-dgx/bin/activate
#   python scripts/run_experiments.py experiments/round2.yaml --bench
#
# TORCH_INDEX selects the CUDA build (default cu130; try cu129/cu128 if the driver is older).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-$REPO/.venv-dgx}"
TORCH_INDEX="${TORCH_INDEX:-cu130}"

python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip
"$VENV/bin/pip" install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
"$VENV/bin/pip" install -r "$REPO/docker/requirements-dgx.txt"
"$VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available: check the driver / TORCH_INDEX"
print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      "capability", torch.cuda.get_device_capability(0))
x = torch.randn(8, 3, 352, 352, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    torch.nn.Conv2d(3, 16, 3, groups=1).cuda()(x).sum().item()
print("bf16 conv OK")
PY
echo "done: source $VENV/bin/activate"
