# Archive: DROID Section 3.1 replication

Moved here on 2026-10-26 with `git mv` (history kept). Not used by the RoboMME pilot in `notebooks/pilot/`.

- `droid_hodor_test_cutie.py`, `inspect_run.py`, `run_all_4090.sh`, `run_section3.1.sh`, `section3.1_session.log`: DROID entry points and logs.
- `configs/`: per-scene DROID configs (giraffe, chicken, pineapple, cabinet, pineapple_llm). `configs/base.yaml` stays at the repo root because the pilot loads it.
- `data/`: DROID recordings. `results/`: earlier run outputs.

The pipeline modules (`post_process_scene_cutie.py`, `vit.py`, `dinov2_encoder.py`, `extract_part_masks_cutie.py`, `stage_timer.py`, `rerun_logger.py`, `llm_entities.py`, `droid_utils.py`) stay at the repo root: the pilot imports `HODORProcessor` from them.

To run the old DROID scripts again, move the files back to the repo root (`git mv archive/droid-section3.1/<file> .`), since they use paths relative to the root.
