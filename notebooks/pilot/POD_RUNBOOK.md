> Historical development record: some commands, paths, pending checks and setup assumptions below describe an earlier state. See the repository README and docs/RESEARCH_STATUS.md for the current status. Preserve this record as provenance.

# Pod runbook: running the pilot notebook on RunPod

The notebook needs Linux, an NVIDIA GPU and working Vulkan, so it runs on a rented pod, not on the Mac.
The chat on the Mac cannot reach the pod. A separate Claude Code session on the pod executes the notebook; paste its outputs back to the Mac chat for review.

## 0. On the Mac: build the upload
`pod_upload.tar.gz` in the repo root holds everything the pod needs: `notebooks/pilot/`, the root pipeline modules, `configs/base.yaml`, `requirements.txt`, `setup_gpu.sh`, the research map and `_skill/build-ipynb/`. Model weights are not included; they are downloaded on the pod.
Rebuild it after any change to these files.

## 1. Create the pod
- GPU: RTX 4090 (24 GB). Template: RunPod PyTorch (includes JupyterLab).
- Volume: 100 GB, mounted at `/workspace`. Only `/workspace` survives a pod stop.
- Edit Template, Environment Variables: `NVIDIA_DRIVER_CAPABILITIES` = `all` (without it the container usually has no graphics driver and ManiSkill cannot render).

## 2. Check Vulkan
Connect, open Jupyter Lab, open a Terminal:
```bash
nvidia-smi --query-gpu=name --format=csv
apt-get update -qq && apt-get install -y -qq libvulkan1 vulkan-tools
vulkaninfo --summary 2>&1 | grep -iE "deviceName|driverName|ERROR"
```
PASS: `deviceName = NVIDIA GeForce RTX 4090` and `driverName = NVIDIA`.
If it fails with `Could not get 'vkCreateInstance' ... for ICD libGLX_nvidia.so.0` (seen on RunPod, driver 595, even with the variable set), point Vulkan at the EGL library instead. Write the fix to `/workspace` so it survives restarts:
```bash
mkdir -p /workspace/env
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.3.0"}}' > /workspace/env/nv_egl_icd.json
cat > /workspace/env/pod_env.sh <<'ENV'
export VK_ICD_FILENAMES=/workspace/env/nv_egl_icd.json
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export PATH="$HOME/.local/bin:$PATH"
ENV
grep -q pod_env.sh ~/.bashrc || echo 'source /workspace/env/pod_env.sh' >> ~/.bashrc
source /workspace/env/pod_env.sh
vulkaninfo --summary 2>&1 | grep -iE "deviceName|driverName|ERROR"
```
After every pod restart: `apt-get install -y -qq libvulkan1 vulkan-tools` and the `grep -q ... ~/.bashrc` line again (both live outside `/workspace`). The notebook's kernel must also see `VK_ICD_FILENAMES`: start Jupyter work from a shell that sourced `pod_env.sh`, or set it in the first notebook cell.

## 3. Upload and unpack
In Jupyter Lab, open `/workspace` in the file panel and upload `pod_upload.tar.gz` with the upload (up-arrow) button. Then:
```bash
mkdir -p /workspace/sg && tar -xzf /workspace/pod_upload.tar.gz -C /workspace/sg --no-same-owner && ls /workspace/sg/notebooks/pilot
```
PASS: the listing shows `sg_interfaces_pilot.ipynb` and `sg_pilot.py`.
Do not run `setup_gpu.sh` as is: it pins `transformers==4.44.2` for GroundingDINO, which conflicts with SmolVLA. The environment plan is made in step 5.

## 4. Install Claude Code on the pod
```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace/sg/.claude/skills && cp -r /workspace/sg/_skill/build-ipynb /workspace/sg/.claude/skills/
cd /workspace/sg && claude
```
First run prints a login URL: open it in a browser on the Mac, sign in, paste the code back into the Terminal.
Claude Code installs outside `/workspace`, so after a pod restart repeat the first two lines.

## 5. Hand off to the pod session
Paste:
```
Use the build-ipynb skill to execute notebooks/pilot/sg_interfaces_pilot.ipynb in smoke mode. Read notebooks/pilot/INSTRUCTIONS.md, BUILD_NOTES.md and _build/outline.md first. The notebook and sg_pilot.py are written but never run. Do not run setup_gpu.sh as is: it pins transformers 4.44.2 for GroundingDINO, which conflicts with SmolVLA; propose an environment plan (likely two venvs) before installing anything. Never modify files outside notebooks/pilot/. Run Step 1 first, show me the version table and the rendered frame, then stop.
```
PASS: it reads the files and proposes an environment plan before installing anything.

## 6. Bring results back
- Paste errors, printed tables and screenshots into the Mac chat.
- Download the executed notebook (right-click it in Jupyter Lab, Download) into `notebooks/pilot/` on the Mac.
- Stop the pod when idle; the volume keeps `/workspace`.
