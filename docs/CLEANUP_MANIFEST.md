# Cleanup manifest — 2026-11-09

## Changes

- Replaced the root README with a collaborator guide, explicit completed/pending status, results locations, dependency limitations and a verified offline check.
- Preserved the previous README verbatim at `archive/documentation/README-before-cleanup.md`; indexed historical documentation at `archive/documentation/README.md`.
- Marked `SECTION_3_1_NOTES.md`, `notebooks/pilot/BUILD_NOTES.md`, `INSTRUCTIONS.md` and `POD_RUNBOOK.md` as historical records; retained their content and original paths. Fixed the obsolete `output/` research-map reference in INSTRUCTIONS.md to `docs/`.
- Removed only `notebooks/pilot/__pycache__/`, generated Python bytecode. No backup is needed: Python regenerates it from source.
- Added `.venv/` and `.ipynb_checkpoints/` ignore rules and `tools/check_repository.py`, a standard-library-only structural check.

## Preserved and recoverable

All report-cited Python and notebook paths are unchanged. All datasets, model weights, logs, figures, executed notebooks, draft paper assets, notebook-building sources and research notes remain in place. The many pre-existing modified DROID H5/MP4 files and untracked HANDOFF.md/docs research materials were not edited or discarded. No uncertain research material was deleted. No external backup was necessary. The old README is recoverable from the archive above; the cleanup is prepared as a narrowly scoped commit; no history was rewritten.

## Validation and limits

The offline check passed: 22 research Python files parsed, five notebooks had valid JSON/basic structure, and key cited paths existed. Shell scripts passed `bash -n`; root README relative links resolved. `git diff --check` passed. This is structural validation only: no model APIs, GPU experiments, installs or training were run. Requirements remain unpinned; the two-environment GPU setup and legacy DROID path assumptions still require care. Historical documents intentionally retain dated commands as provenance.

## GitHub handoff follow-up

Added `docs/RESEARCH_STATUS.md` as the shareable research summary and updated README references to it. The original untracked HANDOFF.md stays unchanged locally because it includes operational volume/session details. Unrelated untracked research notes and paper binaries are excluded. No notebooks, raw logs, data or binaries are newly staged.

Publication scope: `.gitignore`, `README.md`, `SECTION_3_1_NOTES.md`, the three pilot documentation files (`BUILD_NOTES.md`, `INSTRUCTIONS.md`, `POD_RUNBOOK.md`), both `archive/documentation/` Markdown files, this manifest, `docs/RESEARCH_STATUS.md`, and `tools/check_repository.py`.

The proposed Git tree was exported into a clean temporary directory and the offline check, README relative links and shell syntax passed there. The exact staged files were reviewed and scanned for credential patterns without displaying matched values. No credentials were found by that scan; it is not a guarantee about historical repository content.

GitHub API access failed (`error connecting to api.github.com`); a direct connectivity check failed with DNS resolution timeout. Remote HEAD verification also could not complete. The actual default branch cannot be verified, so no default-branch setting was changed and no PR or publication is claimed. After connectivity returns, fetch and review remote divergence before a normal push of `pi05-pilot`, then open a PR to the verified default branch if different. Existing historical commits may contain logs/binaries and must be reviewed if they are not already remote.
