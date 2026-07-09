#!/usr/bin/env python3
"""Write ls-like details for files ending with .gitbackup under a root directory.

Example:
    python scripts/list_gitbackup_files.py

Example with custom output path:
    python scripts/list_gitbackup_files.py --output gitbackup_files.txt

Example with custom search root:
    python scripts/list_gitbackup_files.py --root /path/to/search --output gitbackup_files.txt
"""

from argparse import ArgumentParser, Namespace
from datetime import datetime
from grp import getgrgid
from pathlib import Path
from pwd import getpwuid
from stat import filemode


DEFAULT_ROOT = Path("/sci/labs/michall/roeizucker/token_cls_models2")
DEFAULT_OUTPUT = Path("gitbackup_files.txt")


def find_gitbackup_files(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.rglob("*.gitbackup") if path.is_file())


def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%b %e %H:%M")


def get_user_name(uid: int) -> str:
    return getpwuid(uid).pw_name


def get_group_name(gid: int) -> str:
    return getgrgid(gid).gr_name


def format_ls_line(path: Path) -> str:
    stat_result = path.stat()
    mode = filemode(stat_result.st_mode)
    owner = get_user_name(stat_result.st_uid)
    group = get_group_name(stat_result.st_gid)
    timestamp = format_timestamp(stat_result.st_mtime)
    return (
        f"{mode} {stat_result.st_nlink} {owner} {group} "
        f"{stat_result.st_size} {timestamp} {path}"
    )


def write_lines(paths: list[Path], output_path: Path) -> None:
    output_path.write_text("\n".join(format_ls_line(path) for path in paths) + "\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = find_gitbackup_files(args.root)
    write_lines(paths, args.output)
    print(f"Wrote {len(paths)} entries to {args.output}")


if __name__ == "__main__":
    main()
