"""Execute cells [0, N) of a notebook in the sg-main kernel and write the executed copy.

    source /workspace/env/pod_env.sh
    export JUPYTER_PATH=/workspace/envs/jupyter/share/jupyter
    python tools/run_upto.py sg_interfaces_pilot.ipynb /workspace/logs/run.ipynb 122

Cells run with the notebook folder as working directory. Prints "ALL CELLS OK" or "FAILED" and the
elapsed time; the executed copy is written either way, so the failing cell's traceback is in it
(see tools/digest.py). Run it in a tmux window: nbclient prints nothing per cell.
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
nb = nbformat.read(src, 4)
nb.cells = nb.cells[:n]
nb.metadata["kernelspec"] = {"name": "sg-main", "display_name": "Python (sg-main)", "language": "python"}
t = time.time()
try:
    NotebookClient(
        nb,
        timeout=3600,
        kernel_name="sg-main",
        resources={"metadata": {"path": str(Path(src).resolve().parent)}},
    ).execute()
    print("ALL CELLS OK")
except CellExecutionError:
    print("FAILED")
finally:
    nbformat.write(nb, out)
    print(f"elapsed {time.time() - t:.0f}s")
