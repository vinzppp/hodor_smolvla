#!/usr/bin/env bash
# One-command environment setup for the HODOR Section 3.1 pipeline on a
# Linux + NVIDIA GPU box. Run from the repo root:
#
#     bash setup_gpu.sh
#
# Prerequisite: a working PyTorch with CUDA. requirements.txt lists a bare
# "torch"; if you do not already have a CUDA build, install one first, e.g.
#     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# then re-run this script.
set -euo pipefail

echo "==> Python deps"
pip install -r requirements.txt

echo "==> GroundingDINO"
[ -d GroundingDINO ] || git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e GroundingDINO

echo "==> segment-anything"
pip install "git+https://github.com/facebookresearch/segment-anything.git"

echo "==> Cutie (do NOT pip install cutie from PyPI - unrelated package)"
[ -d Cutie ] || git clone https://github.com/hkchengrex/Cutie.git
pip install -e Cutie

echo "==> SAM ViT-H checkpoint (~2.6GB)"
mkdir -p pretrained
if [ ! -f pretrained/sam_vit_h_4b8939.pth ]; then
  wget -P pretrained https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
  echo "    already present, skipping"
fi

echo "==> Prefetch DINOv2 ViT-B/14 (optional, for the faithful backbone)"
python - <<'PY' || echo "    DINOv2 prefetch skipped (fetch it on first run instead)"
import torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
print("    DINOv2 cached")
PY

echo
echo "Setup complete. Sanity checks:"
echo "  python -c 'import groundingdino, segment_anything, cutie; print(\"perception deps OK\")'"
echo
echo "Baseline run (existing fixed-class pipeline):"
echo "  python droid_hodor_test_cutie.py --config configs/giraffe.yaml"
echo
echo "Faithful variants:"
echo "  export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY"
echo "  python llm_entities.py --config configs/pineapple_llm.yaml --write   # inspect entities"
echo "  python droid_hodor_test_cutie.py --config configs/pineapple_llm.yaml # LLM entities"
echo "  python droid_hodor_test_cutie.py --config configs/giraffe_dinov2.yaml # DINOv2 backbone"
