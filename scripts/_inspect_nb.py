"""Print code-cell outputs from an executed notebook (for sanity checks)."""
import sys
from pathlib import Path

import nbformat

# Force UTF-8 stdout so we can print math symbols on Windows consoles
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = Path(sys.argv[1])
nb = nbformat.read(path, as_version=4)

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    print(f"\n=== code cell {i} ===")
    for out in cell.get("outputs", []):
        t = out.get("output_type")
        if t == "stream":
            print(out.get("text", ""), end="")
        elif t in ("execute_result", "display_data"):
            data = out.get("data", {})
            txt = data.get("text/plain", "")
            if "image/png" in data:
                txt = txt + "  [image png omitted]"
            print(txt)
        elif t == "error":
            print("ERROR:", out.get("ename"), out.get("evalue"))
            for line in out.get("traceback", []):
                print(line)
