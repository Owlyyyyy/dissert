from __future__ import annotations

import argparse
from pathlib import Path

from .preview_rebuild import rebuild_preview_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a preview folder using verified track-level artists")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--old-preview-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rebuild_preview_folder(
        Path(args.source_manifest), Path(args.old_preview_folder),
        Path(args.output_folder), Path(args.output_manifest), Path(args.env_file), args.limit,
    )


if __name__ == "__main__":
    main()
