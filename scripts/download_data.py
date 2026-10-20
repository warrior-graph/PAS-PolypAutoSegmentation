"""Download the standard polyp benchmark (PraNet / Polyp-PVT split) into data/.

    python scripts/download_data.py [--root data]

TrainDataset: Kvasir-SEG (900) + CVC-ClinicDB (550)
TestDataset:  Kvasir (100), CVC-ClinicDB (62), CVC-ColonDB (380), ETIS-LaribPolypDB (196), CVC-300 (60)
"""

import argparse
import shutil
import zipfile
from pathlib import Path

FILES = {
    "TrainDataset": "1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb",
    "TestDataset": "1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao",
}

MANUAL = """
Automatic download failed. Download manually and unzip into {root}/:
  TrainDataset: https://drive.google.com/file/d/{TrainDataset}/view
  TestDataset:  https://drive.google.com/file/d/{TestDataset}/view
(Links from the PraNet repository: https://github.com/DengPingFan/PraNet)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        raise SystemExit("pip install gdown  (or see the manual links in this script)")

    for name, file_id in FILES.items():
        if (root / name).is_dir():
            print(f"{root / name} already exists, skipping")
            continue
        archive = root / f"{name}.zip"
        if not archive.exists() and gdown.download(id=file_id, output=str(archive), quiet=False) is None:
            raise SystemExit(MANUAL.format(root=root, **FILES))
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            zf.extractall(root, members)
        if not args.keep_zip:
            archive.unlink()
    shutil.rmtree(root / "__MACOSX", ignore_errors=True)
    for name in FILES:
        n = sum(1 for _ in (root / name).rglob("*.png"))
        print(f"{root / name}: {n} png files")


if __name__ == "__main__":
    main()
