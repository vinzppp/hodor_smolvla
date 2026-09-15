#!/usr/bin/env bash
# =============================================================================
# HODOR Section 3.1 (Task-Driven Scene Graph) — full reproduction runbook
# NOTE: superseded by run_all_4090.sh (one-shot, logged). Kept for reference.
# From a fresh RunPod A40 48GB pod (torch 2.8 / CUDA 12.8 / Python 3.12 image)
# to all three validated pipelines.
#
# Read and run it step by step. Steps 3/5/7 are long (10-25 min each), so this
# is a runbook, not a fire-and-forget script — especially because of the tokens
# in step 0. Nothing secret is committed to the repo.
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# 0. Tokens. Paste your own values. These live only in this shell session;
#    do NOT hard-code them into any file that gets committed.
#      - GH_PAT: GitHub personal access token with 'repo' scope (private repo).
#      - OPENROUTER_API_KEY: used later by the LLM entity front-end (step 5).
# -----------------------------------------------------------------------------
export GH_PAT=ghp_...                    # <-- your GitHub token
export OPENROUTER_API_KEY=sk-or-...      # <-- your OpenRouter key

# -----------------------------------------------------------------------------
# 1. Get the code + data onto the pod (Git-LFS aware).
#    Why two repos: the big files (data/, pretrained/) are stored with Git-LFS.
#    Your repo vinzppp/scene-graph has all your CODE but only LFS *pointer*
#    files for the big blobs (your Mac had no git-lfs when you pushed). So we
#    clone the upstream repo for the REAL LFS blobs, then overlay your code on
#    top — overlaying only code paths keeps upstream's real data/weights intact.
# -----------------------------------------------------------------------------
apt-get update && apt-get install -y git-lfs     # image may not ship git-lfs
git lfs install

cd /workspace
git clone https://github.com/jianingq/scene-graph-robot.git
cd scene-graph-robot                              # real data/ and pretrained/ (LFS)

# Add your private repo and fetch your branch of changes.
git remote add fork https://${GH_PAT}@github.com/vinzppp/scene-graph.git
git fetch fork
git branch -r                                     # find your branch under fork/...

# Overlay ONLY your code/config files from your branch (default branch: main).
# This pulls in llm_entities.py, dinov2_encoder.py, your edits to vit.py /
# post_process_scene_cutie.py, the configs, and setup_gpu.sh — WITHOUT touching
# upstream's real LFS data/ and pretrained/ blobs.
git checkout fork/main -- '*.py' configs setup_gpu.sh SECTION_3_1_NOTES.md

# -----------------------------------------------------------------------------
# 2. One-shot environment setup.
#    setup_gpu.sh installs everything and bakes in every fix needed on the
#    torch-2.8 / py3.12 image:
#      - pip install -r requirements.txt
#      - GroundingDINO: patch ms_deform_attn_cuda.cu for torch 2.8
#          (.scalar_type().is_cuda() -> .is_cuda(),  .data<T>() -> .data_ptr<T>())
#        then build with --no-build-isolation
#      - segment-anything, Cutie
#      - NLP-stack pin that fixes GroundingDINO's BertModelWarper crash
#          (transformers 5.x dropped BertModel.get_extended_attention_mask):
#          transformers==4.44.2 tokenizers==0.19.1 huggingface_hub==0.25.2 timm==0.9.16
#      - SAM ViT-H checkpoint (~2.4 GB)
#      - DINOv2 ViT-B/14 prefetch
# -----------------------------------------------------------------------------
bash setup_gpu.sh

# 2a. Sanity check: the perception deps import cleanly.
python -c 'import groundingdino, segment_anything, cutie; print("perception deps OK")'

# -----------------------------------------------------------------------------
# 3. Manual-class pipeline (DINOv2 backbone, hand-specified classes)
#    detect -> segment -> embed -> track over all 10 giraffe episodes, writing
#    episode_pr.npy / episode_bboxes.npy per episode. (~25 min)
# -----------------------------------------------------------------------------
python droid_hodor_test_cutie.py --config configs/giraffe.yaml

# 3a. Inspect the baseline arrays.
#     Expect pr (T, N+1, 768) and bboxes (T, N+1, 7); slot 0 = global scene node.
python -c "
import numpy as np, glob
pr = sorted(glob.glob('data/giraffe/*/episode_pr.npy'))[0]
bb = pr.replace('episode_pr.npy','episode_bboxes.npy')
a, b = np.load(pr), np.load(bb)
print('file:', pr)
print('pr    ', a.shape, a.dtype)
print('bboxes', b.shape, b.dtype)
"

# -----------------------------------------------------------------------------
# 4. Faithful variant A — LLM entity front-end.
#    The LLM reads the task string and drives GroundingDINO's classes instead of
#    them being hand-typed into the yaml. (OPENROUTER_API_KEY set in step 0.)
# -----------------------------------------------------------------------------
# 4a. Preview what the LLM extracts, and cache it (--write). For
#     "pick up the pineapple and place it in the bowl" this yields
#     entities ["pineapple","bowl"] -> classes ["pineapple.","bowl."].
python llm_entities.py --config configs/pineapple_llm.yaml --write

# 4b. Run the full pipeline with the LLM-derived classes. (~10 min)
python droid_hodor_test_cutie.py --config configs/pineapple_llm.yaml

# 4c. Inspect the outputs. Slot count = scene node + pineapple + bowl = 3.
python -c "
import numpy as np, glob
pr = sorted(glob.glob('data/pineapple/*/episode_pr.npy'))[0]
bb = pr.replace('episode_pr.npy','episode_bboxes.npy')
a, b = np.load(pr), np.load(bb)
print('file:', pr)
print('pr    ', a.shape, a.dtype)
print('bboxes', b.shape, b.dtype)
"

# (Step 5, the separate DINOv2 config, was removed: DINOv2 is now the default
#  backbone in configs/base.yaml, so steps 3 and 4 already use it.)

echo "Section 3.1 pipelines complete (DINOv2)."
