from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from vehicle import Vehicle
from config import GRID_SIZE


LEVELS_FILE = "levels.txt"


def load_level(level_number: int) -> List[Vehicle]:
    # level_number is 1-based
    if level_number < 1:
        raise ValueError("level_number must be >= 1")

    levels = _read_levels_from_file(Path(LEVELS_FILE))
    idx = level_number - 1

    if idx >= len(levels):
        raise ValueError(f"Requested level {level_number}, only {len(levels)} available")

    return _parse_grid_lines(levels[idx]["grid"])


def list_levels() -> int:
    return len(_read_levels_from_file(Path(LEVELS_FILE)))


def get_level_meta(level_number: int) -> Dict[str, str]:
    levels = _read_levels_from_file(Path(LEVELS_FILE))
    idx = level_number - 1
    if idx < 0 or idx >= len(levels):
        raise ValueError("Invalid level_number")
    return levels[idx]["meta"]


def _read_levels_from_file(path: Path) -> List[Dict[str, object]]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()

    results: List[Dict[str, object]] = []
    current_grid: List[str] = []
    current_meta: Dict[str, str] = {}

    def flush_block() -> None:
        nonlocal current_grid, current_meta
        if not current_grid:
            current_meta = {}
            return

        if len(current_grid) != GRID_SIZE:
            raise ValueError(f"{path}: block has {len(current_grid)} lines, expected {GRID_SIZE}")

        for i, row in enumerate(current_grid, start=1):
            if len(row) != GRID_SIZE:
                raise ValueError(f"{path}: grid line {i} length {len(row)}, expected {GRID_SIZE}")

        results.append({"grid": current_grid, "meta": current_meta})
        current_grid = []
        current_meta = {}

    for line in raw_lines:
        s = line.strip()

        if not s:
            flush_block()
            continue

        if s.startswith("#"):
            # Header line, optional format: "# key: value | key2: value2"
            # Or "# 12 | Beginner"
            header = s[1:].strip()
            if header:
                current_meta["header"] = header
            continue

        current_grid.append(s)

    flush_block()
    return results


def _parse_grid_lines(lines: List[str]) -> List[Vehicle]:
    cells: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            ch = lines[r][c]
            if ch == ".":
                continue
            cells[ch].append((r, c))

    vehicles: List[Vehicle] = []

    for vid, coords in cells.items():
        coords.sort()
        rs = sorted({r for r, _ in coords})
        cs = sorted({c for _, c in coords})
        size = len(coords)

        if size < 2:
            raise ValueError(f"Vehicle {vid} has size {size}, expected 2 or 3")

        if len(rs) == 1 and len(cs) == size:
            row = rs[0]
            col0 = cs[0]
            if cs != list(range(col0, col0 + size)):
                raise ValueError(f"Vehicle {vid} not contiguous horizontally: {coords}")
            vehicles.append(Vehicle(vid, "h", size, row, col0))

        elif len(cs) == 1 and len(rs) == size:
            col = cs[0]
            row0 = rs[0]
            if rs != list(range(row0, row0 + size)):
                raise ValueError(f"Vehicle {vid} not contiguous vertically: {coords}")
            vehicles.append(Vehicle(vid, "v", size, row0, col))

        else:
            raise ValueError(f"Vehicle {vid} invalid shape/orientation: {coords}")

    _validate_rush_hour_ids(vehicles)
    return vehicles


def _validate_rush_hour_ids(vehicles: List[Vehicle]) -> None:
    seen = set()
    for v in vehicles:
        if v.id in seen:
            raise ValueError(f"Duplicate vehicle id: {v.id}")
        seen.add(v.id)

        if v.id == "X":
            if v.size != 2 or v.orientation != "h":
                raise ValueError("X must be a horizontal size-2 car")
            continue

        if "A" <= v.id <= "K":
            if v.size != 2:
                raise ValueError(f"{v.id} must be size 2 (car), got {v.size}")
        elif "O" <= v.id <= "R":
            if v.size != 3:
                raise ValueError(f"{v.id} must be size 3 (truck), got {v.size}")
        else:
            # Allow other ids if present
            pass
