"""Shared helpers for building notebooks via nbformat."""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    # dedent: strip leading spaces uniformly from triple-quoted blocks
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    indent = min(
        (len(l) - len(l.lstrip(" ")) for l in lines if l.strip()),
        default=0,
    )
    text = "\n".join(l[indent:] if len(l) >= indent else l for l in lines)
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    indent = min(
        (len(l) - len(l.lstrip(" ")) for l in lines if l.strip()),
        default=0,
    )
    text = "\n".join(l[indent:] if len(l) >= indent else l for l in lines)
    return nbf.v4.new_code_cell(text)


def build_notebook(cells, out_path: Path) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = list(cells)
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out_path)
    return out_path
