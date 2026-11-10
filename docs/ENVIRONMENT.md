# Recorded pilot environment

The RoboMME notebooks used Linux, an NVIDIA GPU and Vulkan. Run the policy notebook and perception worker in separate Python environments because they require different versions of Transformers.

- Policy notebook: Python 3.12, PyTorch 2.9.1+cu128, LeRobot commit `5831498`, Transformers 5.5.4, PEFT 0.18.1, RoboMME commit `53673b3` and its ManiSkill 3.0.0b21 fork. The pilot used NumPy 1.26.4 and Gymnasium 0.29.1.
- Perception worker: Python 3.12, PyTorch 2.9.1, Transformers 4.44.2, Tokenizers 0.19.1, Hugging Face Hub 0.25.2, timm 0.9.16, OpenCV below 5 and NumPy 2.2.

Execution requires RoboMME replay data and the GroundingDINO, SAM, Cutie and DINOv2 model files. See the [research status](RESEARCH_STATUS.md) for the scope of the recorded runs.
