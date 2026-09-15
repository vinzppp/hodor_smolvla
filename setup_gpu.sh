#!/usr/bin/env bash
# =============================================================================
# setup_gpu.sh — build the HODOR Section 3.1 perception environment.
#
# Idempotent + cache-aware: re-running skips anything already done, and all
# model weights are cached on the persistent /workspace volume. So the FIRST
# pod takes ~10 min, but any restarted pod (same network volume) is ready in
# seconds because nothing needs to recompile or re-download.
#
# It writes /workspace/hodor_env.sh — source that at the start of every shell
# (including before the pipeline runs) so the runs use the same persistent
# caches instead of re-downloading BERT/ResNet/Cutie weights.
# =============================================================================
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

# --- Persistent caches on /workspace (survive pod stop/restart) --------------
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# --- Compile GroundingDINO for THIS GPU's arch only (A40/A6000/3090 = 8.6) ---
# Without this, nvcc builds the CUDA kernel for every arch = a much slower build.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

# Persist the env vars for future shells / the pipeline runs.
cat > /workspace/hodor_env.sh <<EOF
export HF_HOME="$HF_HOME"
export TORCH_HOME="$TORCH_HOME"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
EOF

have() { python -c "import $1" 2>/dev/null; }

echo "==> Python deps"
pip install -q -r requirements.txt

echo "==> GroundingDINO"
if python -c "import groundingdino._C" 2>/dev/null; then
  echo "    already built, skipping"
else
  [ -d GroundingDINO ] || git clone https://github.com/IDEA-Research/GroundingDINO.git
  CU=GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu
  # torch-2.8 kernel patches (idempotent — safe to re-run):
  sed -i 's/value\.type()\.is_cuda()/value.is_cuda()/g' "$CU"   # assert
  sed -i 's/value\.type()/value.scalar_type()/g'        "$CU"   # AT_DISPATCH
  sed -i 's/\.data<\([^>]*\)>()/.data_ptr<\1>()/g'      "$CU"   # accessor
  pip install -q --no-build-isolation -e GroundingDINO
fi

echo "==> segment-anything"
have segment_anything || pip install -q git+https://github.com/facebookresearch/segment-anything.git

echo "==> Cutie (do NOT pip install cutie from PyPI - unrelated package)"
if ! have cutie; then
  [ -d Cutie ] || git clone https://github.com/hkchengrex/Cutie.git
  pip install -q -e Cutie
fi

echo "==> NLP stack pin (transformers<4.45 for GroundingDINO's BertModelWarper)"
if ! python -c "import transformers,sys; sys.exit(0 if transformers.__version__=='4.44.2' else 1)" 2>/dev/null; then
  pip install -q --no-deps "transformers==4.44.2" "tokenizers==0.19.1" \
      "huggingface_hub==0.25.2" "timm==0.9.16"
fi

echo "==> SAM ViT-H checkpoint (~2.4GB)"
mkdir -p pretrained
if [ ! -f pretrained/sam_vit_h_4b8939.pth ]; then
  wget -q --show-progress -O pretrained/sam_vit_h_4b8939.pth \
      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
  echo "    already present, skipping"
fi

echo "==> Prefetch DINOv2 ViT-B/14 (into persistent TORCH_HOME)"
python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitb14')" >/dev/null 2>&1 \
  && echo "    DINOv2 cached" || echo "    DINOv2 prefetch skipped (will load on first run)"

echo
echo "Setup complete. Before running the pipeline, load the persistent caches:"
echo "  source /workspace/hodor_env.sh"
echo "Sanity check:"
echo "  python -c 'import groundingdino, segment_anything, cutie; print(\"perception deps OK\")'"
