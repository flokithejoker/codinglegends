"""Download ICD-10 XML/PDF releases locally so the dataloaders can read code descriptions."""

import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from icd_codes.connectors.cms import download_cms_icd_version


def main() -> None:
    parser = ArgumentParser(description="Fetch ICD assets into data/medical-coding-systems/icd")
    parser.add_argument("--year", type=int, default=2025, help="ICD release year to download (default: 2025)")
    parser.add_argument(
        "--use-update",
        action="store_true",
        help="Prefer month-tagged updated releases when available",
    )
    args = parser.parse_args()

    target_dir = Path("data/medical-coding-systems/icd")
    target_dir.mkdir(parents=True, exist_ok=True)

    download_cms_icd_version(target_dir, args.year, use_update=args.use_update)


if __name__ == "__main__":
    main()
