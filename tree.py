"""Create a bounded project tree and a text-file content snapshot.

Run ``python tree.py`` from any directory to scan the project root containing
this file. The generated files are intentionally excluded from the snapshot.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MAX_DEPTH = 6
DEFAULT_TREE_FILE = "tree_str"
DEFAULT_CONTENT_FILE = "tree.txt"
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "env",
    "node_modules",
    "tree",
    "venv",
}
EXCLUDED_FILES = {"tree.txt", "tree_str", "tree_str.txt"}
MAX_TEXT_FILE_BYTES = 1_000_000


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the tree generator."""
    parser = argparse.ArgumentParser(
        description="Write a bounded directory tree and text-file snapshot."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to scan (default: the directory containing tree.py).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"Maximum depth below root to include (default: {DEFAULT_MAX_DEPTH}).",
    )
    parser.add_argument(
        "--tree-file",
        type=Path,
        default=Path(DEFAULT_TREE_FILE),
        help=f"Tree output path (default: {DEFAULT_TREE_FILE}).",
    )
    parser.add_argument(
        "--content-file",
        type=Path,
        default=Path(DEFAULT_CONTENT_FILE),
        help=f"Content output path (default: {DEFAULT_CONTENT_FILE}).",
    )
    args = parser.parse_args()
    if args.max_depth < 0:
        parser.error("--max-depth must be zero or greater")
    return args


def relative_path(path: Path, root: Path) -> str:
    """Return a stable, platform-independent relative path."""
    return path.relative_to(root).as_posix()


def should_skip(path: Path, root: Path, output_paths: set[Path]) -> bool:
    """Return whether a path should be omitted from both generated files."""
    if path in output_paths:
        return True
    if path.name in EXCLUDED_FILES:
        return True
    return any(part in EXCLUDED_DIRECTORIES for part in path.relative_to(root).parts)


def iter_entries(
    directory: Path, root: Path, depth: int, max_depth: int, output_paths: set[Path]
) -> list[Path]:
    """Return visible entries in deterministic order."""
    if depth > max_depth:
        return []

    entries: list[Path] = []
    for entry in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold())):
        if should_skip(entry, root, output_paths):
            continue
        if entry.is_dir() and depth < max_depth:
            entries.append(entry)
        elif entry.is_file():
            entries.append(entry)
    return entries


def render_tree(
    directory: Path, root: Path, depth: int, max_depth: int, output_paths: set[Path]
) -> list[str]:
    """Render a bounded tree using the familiar ``tree /F`` branch style."""
    lines: list[str] = []
    entries = iter_entries(directory, root, depth, max_depth, output_paths)
    for index, entry in enumerate(entries):
        branch = "└── " if index == len(entries) - 1 else "├── "
        lines.append(f"{'│   ' * depth}{branch}{entry.name}")
        if entry.is_dir() and depth < max_depth:
            lines.extend(render_tree(entry, root, depth + 1, max_depth, output_paths))
    return lines


def is_text_file(path: Path) -> bool:
    """Identify reasonably sized UTF-8 or plain-text files."""
    if path.stat().st_size > MAX_TEXT_FILE_BYTES:
        return False
    sample = path.read_bytes()[:8192]
    return b"\x00" not in sample


def render_content(
    directory: Path, root: Path, depth: int, max_depth: int, output_paths: set[Path]
) -> list[str]:
    """Collect readable text-file contents from the bounded tree."""
    lines: list[str] = []
    for entry in iter_entries(directory, root, depth, max_depth, output_paths):
        if entry.is_dir():
            lines.extend(render_content(entry, root, depth + 1, max_depth, output_paths))
            continue
        if not is_text_file(entry):
            continue
        try:
            content = entry.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lines.append(f"\n{'=' * 80}\nFILE: {relative_path(entry, root)}\n{'=' * 80}\n")
        lines.append(content.rstrip())
        lines.append("\n")
    return lines


def main() -> None:
    """Generate the tree and content snapshot."""
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist: {root}")

    tree_path = args.tree_file.resolve()
    content_path = args.content_file.resolve()
    output_paths = {tree_path, content_path}
    tree_lines = [root.name, *render_tree(root, root, 0, args.max_depth, output_paths)]
    content_lines = render_content(root, root, 0, args.max_depth, output_paths)

    tree_path.write_text("\n".join(tree_lines) + "\n", encoding="utf-8")
    content_path.write_text("".join(content_lines), encoding="utf-8")
    print(f"Wrote {tree_path}")
    print(f"Wrote {content_path}")


if __name__ == "__main__":
    main()
