# Research status — 2026-11-09

This shareable summary records the completed work and limitations from the project's development handoff. Operational account, volume and chat-session details remain local; they are not required to read this repository.

## Completed evidence

- HODOR Section 3.1 perception was replicated with GroundingDINO, SAM, Cutie and DINOv2. Root modules and `archive/droid-section3.1/` preserve implementation and earlier evidence. Dynamic node insertion and full HODOR policy replication are not completed.
- SmolVLA trained on 100 single-task VideoUnmaskSwap demonstrations, succeeded on training examples and reached approximately 4% test success. The recipe used a high learning rate and executed 50-action chunks open-loop. These are limitations and suspected contributors, not isolated causal findings.
- The π0.5 port passed 35 smoke checks; A0–A3 each trained for ten steps and A2/A3 exercised closed-loop evaluation. Full training and controlled comparisons remain pending.
- Memory diagnostics over 100 training episodes found 62 with frozen-rule error above 10 pixels at the decision frame. The bound rule follows the covering container; its recorded median error was 2.9 pixels, maximum 5.0 pixels. These are graph diagnostics, not policy success rates.
- On episode 0, recorded slot identity accuracy was 0.983 over 719 post-swap slot-frames. Cutie can keep a hidden cube's slot on its covering container, making the perceived graph unusually close to oracle on this task. This limits generalization claims.

## Reading and reproduction

Start with [the π0.5 notebook](../notebooks/pilot/sg_interfaces_pi05.ipynb), [executed smoke output](../notebooks/pilot/executed/sg_interfaces_pi05_smoke.ipynb) and the [recorded environment](ENVIRONMENT.md). The [SmolVLA notebook](../notebooks/pilot/sg_interfaces_pilot.ipynb) remains part of the research history. Logs and figures remain under `notebooks/pilot/`; historical DROID results remain under `archive/droid-section3.1/results/`.

The Linux GPU setup uses separate policy and perception environments because their transformers versions conflict. Full RoboMME H5 data, replay caches and trained checkpoints may require separate access from the maintainer. The repository's offline structural check does not reproduce GPU results. Notebook placeholders and proposed experiment grids are not measured findings.

Next work is a π0.5 baseline sanity gate, multi-task data preparation and full controlled training/evaluation before making interface-performance claims. No paid compute or new experiments were performed for repository cleanup.
