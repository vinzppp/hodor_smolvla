"""Offline structural smoke check; no imports of research code or model downloads."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
python_files = list(ROOT.glob('*.py')) + list((ROOT / 'notebooks/pilot').rglob('*.py'))
for path in python_files:
    ast.parse(path.read_text(), filename=str(path))
notebooks = list((ROOT / 'notebooks/pilot').rglob('*.ipynb'))
for path in notebooks:
    nb = json.loads(path.read_text())
    assert nb['nbformat'] == 4 and isinstance(nb['cells'], list), path
    for cell in nb['cells']:
        assert cell['cell_type'] in {'code', 'markdown', 'raw'}, path
        assert isinstance(cell['source'], (str, list)), path
for name in ('post_process_scene_cutie.py', 'extract_part_masks_cutie.py', 'dinov2_encoder.py',
             'notebooks/pilot/sg_pilot.py', 'notebooks/pilot/pi05_policy.py',
             'notebooks/pilot/executed/sg_interfaces_pi05_smoke.ipynb'):
    assert (ROOT / name).is_file(), name
print(f'PASS: {len(python_files)} Python files parse; {len(notebooks)} notebooks have valid JSON/basic structure; cited code paths exist.')
print('This does not validate dependencies, notebook execution, models, or scientific results.')

for target in re.findall(r"\]\(([^)]+)\)", (ROOT / "README.md").read_text()):
    if "://" not in target and not target.startswith("#"):
        assert (ROOT / target.split("#")[0]).exists(), f"Missing README target: {target}"
print("PASS: all README relative links resolve.")
