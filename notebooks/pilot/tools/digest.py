"""Print the outputs of an executed notebook as text, and save its images.

    python tools/digest.py <image dir> <executed.ipynb> [first cell index]
    python tools/digest.py /workspace/logs/img /workspace/logs/run.ipynb 57 | grep -E "PASS|FAIL|ERROR"

Streams are printed per code cell; errors with their traceback; images are written to
<image dir>/img_c<cell>.png and their path printed.
"""

import base64
import re
import sys
from pathlib import Path

import nbformat

img_dir, path = Path(sys.argv[1]), sys.argv[2]
lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
img_dir.mkdir(parents=True, exist_ok=True)
nb = nbformat.read(path, 4)


def strip(x):
    return re.sub(r"\x1b\[[0-9;]*m", "", x)


for i, c in enumerate(nb.cells):
    if c.cell_type != "code" or i < lo:
        continue
    print(f"=== cell {i} (exec {c.execution_count}) {c.source.splitlines()[0] if c.source else ''}")
    for o in c.outputs:
        if o.output_type == "stream":
            print(f"[{o.name}]", strip(o.text).rstrip()[-3000:])
        elif o.output_type == "error":
            print("ERROR", o.ename, strip(o.evalue))
            print(strip("\n".join(o.traceback))[-3500:])
        elif "image/png" in o.get("data", {}):
            f = img_dir / f"img_c{i}.png"
            f.write_bytes(base64.b64decode(o.data["image/png"]))
            print("<image>", f)
        elif "text/plain" in o.get("data", {}):
            print(o.data["text/plain"][:300])
