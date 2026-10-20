#!/usr/bin/env bash
# Run PAS inside the NGC PyTorch container on a DGX Spark (or any NVIDIA box with Docker).
#
#   scripts/dgx/run.sh check                      # GPU, versions, tests
#   scripts/dgx/run.sh bench                      # time a short B0 run (~3 min)
#   scripts/dgx/run.sh matrix [runner args...]    # experiments/round2.yaml, e.g. --only e1_b0_strong
#   scripts/dgx/run.sh shell                      # interactive shell in the container
#   scripts/dgx/run.sh -- python train.py ...     # any command
#
# Long runs survive a closed SSH session with:  nohup scripts/dgx/run.sh matrix > dgx.log 2>&1 &
# Environment: IMAGE (default pas-dgx), BASE (NGC tag), REBUILD=1 to rebuild the image,
#              DATA_DIR (host dir with TrainDataset/TestDataset, mounted at /data; default: <repo>/data,
#              downloaded on first use), MATRIX (default experiments/round2.yaml).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-pas-dgx}"
DATA_ROOT=data
MATRIX="${MATRIX:-experiments/round2.yaml}"

if [[ "${REBUILD:-0}" == 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> building $IMAGE"
  docker build -t "$IMAGE" ${BASE:+--build-arg BASE="$BASE"} -f "$REPO/docker/Dockerfile.dgx" "$REPO"
fi

mkdir -p "$REPO/.cache/home"
DOCKER_ARGS=(
  --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
  --user "$(id -u):$(id -g)"
  -e HOME=/workspace/PAS/.cache/home -e HF_HOME=/workspace/PAS/.cache/hf
  -e TORCH_HOME=/workspace/PAS/.cache/torch -e PYTHONUNBUFFERED=1
  -v "$REPO:/workspace/PAS" -w /workspace/PAS
)
if [[ -n "${DATA_DIR:-}" ]]; then
  DOCKER_ARGS+=(-v "$(cd "$DATA_DIR" && pwd):/data"); DATA_ROOT=/data
fi
[[ -t 0 && -t 1 ]] && DOCKER_ARGS+=(-it)

in_container() { docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -lc "$*"; }
ensure_data() { echo "python scripts/download_data.py --root $DATA_ROOT"; }

cmd="${1:-matrix}"; shift || true
case "$cmd" in
  check)
    in_container "nvidia-smi; python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('gpu', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0)); \
from pas.utils import amp_dtype; print('autocast dtype', amp_dtype(torch.device('cuda')))\" && python -m pytest -q tests" ;;
  bench)
    in_container "$(ensure_data) && python scripts/run_experiments.py $MATRIX --data-root $DATA_ROOT --bench $*" ;;
  matrix)
    in_container "$(ensure_data) && python scripts/run_experiments.py $MATRIX --data-root $DATA_ROOT $*" ;;
  shell)
    in_container "bash" ;;
  --)
    in_container "$*" ;;
  *)
    echo "unknown command: $cmd (check | bench | matrix | shell | -- <cmd>)" >&2; exit 2 ;;
esac
