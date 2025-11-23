from __future__ import annotations

import re
from pathlib import Path


def test_selection_range_imports_are_contained() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src" / "tinkerbell"
    allowed_files = {
        (src_root / "ai" / "tools" / "selection_range.py").resolve(),
    }
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        if path.resolve() in allowed_files:
            continue
        relative = path.relative_to(project_root)
        if "editor" in relative.parts:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if re.match(r"\s*(from|import).*\bSelectionRange\b", line):
                    offenders.append(f"{relative}:{line_no}: {line.strip()}")
    assert not offenders, "SelectionRange should only be imported inside editor/* or ai/tools/selection_range.py\n" + "\n".join(offenders)


def test_selection_utils_module_stays_removed() -> None:
    project_root = Path(__file__).resolve().parents[1]
    selection_utils = project_root / "src" / "tinkerbell" / "ai" / "selection_utils.py"
    assert not selection_utils.exists(), "selection_utils.py must remain deleted; reintroduce span helpers inline instead"

    offenders: list[str] = []
    for path in (project_root / "src").rglob("*.py"):
        relative = path.relative_to(project_root)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if re.match(r"\s*from\s+.+\bselection_utils\b", line) or re.match(r"\s*import\s+.+\bselection_utils\b", line):
                    offenders.append(f"{relative}:{line_no}: {line.strip()}")
    assert not offenders, "selection_utils module must not be imported anywhere under src/\n" + "\n".join(offenders)
