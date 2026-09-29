"""Download the login-free FlyWire v783 files used by the 'public' ingestion source (~130 MB).

    python -m backend.preprocessing.download
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from backend.preprocessing.ingest import PUBLIC_ANNOTATIONS, PUBLIC_CONNECTIVITY
from backend.utils.config import RAW_PUBLIC_DIR

PUBLIC_URLS = {
    # FlyWire neuron annotations, Schlegel et al. 2024 (Nature), root IDs of release v783
    PUBLIC_ANNOTATIONS: "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    # v783 neuron-to-neuron synapse counts, Shiu et al. 2024 (Nature)
    PUBLIC_CONNECTIVITY: "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
}


def download_public(raw_dir: Path = RAW_PUBLIC_DIR, force: bool = False) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in PUBLIC_URLS.items():
        target = raw_dir / name
        if target.exists() and not force:
            print(f"skip {name} (already {target.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"downloading {name} ...")
            partial = target.with_name(target.name + ".part")
            urllib.request.urlretrieve(url, partial)
            partial.replace(target)
            print(f"  saved {target.stat().st_size / 1e6:.1f} MB")
        paths.append(target)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    download_public(force=parser.parse_args().force)
