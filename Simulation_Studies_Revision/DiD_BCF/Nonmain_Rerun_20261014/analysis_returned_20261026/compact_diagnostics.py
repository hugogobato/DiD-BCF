"""Losslessly store the collected scalar diagnostics, without changing values."""
from pathlib import Path
import gzip
import hashlib
import json
import os

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "Results/collected_20261026/diagnostics.csv"
TARGET = SOURCE.with_suffix(".csv.gz")
RECORD = SOURCE.parent / "diagnostics_storage.json"


def digest(path, compressed=False):
    h = hashlib.sha256()
    size = 0
    newlines = 0
    reader = gzip.open if compressed else open
    with reader(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
            size += len(chunk)
            newlines += chunk.count(bytes([10]))
    return h.hexdigest(), size, newlines


def main():
    if not SOURCE.exists():
        saved = json.loads(RECORD.read_text())
        assert digest(TARGET, compressed=True) == (
            saved["uncompressed_sha256"], saved["uncompressed_bytes"], saved["newline_count"]
        )
        assert digest(TARGET)[0] == saved["compressed_sha256"]
        print("Existing lossless diagnostics archive verified.")
        return
    original = digest(SOURCE)
    temporary = TARGET.with_suffix(".gz.tmp")
    try:
        with temporary.open("wb") as destination:
            with gzip.GzipFile(filename="", fileobj=destination, mode="wb", compresslevel=6, mtime=0) as packed:
                with SOURCE.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(2**20), b""):
                        packed.write(chunk)
        assert digest(temporary, compressed=True) == original, "Lossless roundtrip failed"
        os.replace(temporary, TARGET)
        record = {
            "status": "PASS", "representation": "lossless gzip of scalar diagnostics CSV",
            "uncompressed_name": SOURCE.name, "compressed_name": TARGET.name,
            "uncompressed_sha256": original[0], "uncompressed_bytes": original[1],
            "newline_count": original[2], "compressed_sha256": digest(TARGET)[0],
            "compressed_bytes": TARGET.stat().st_size,
            "values_or_rows_removed": False, "posterior_arrays_present": False,
            "read_with": "pandas.read_csv('diagnostics.csv.gz') or gzip.open",
        }
        RECORD.write_text(json.dumps(record, indent=2) + "\n")
        # Every byte is recoverable from TARGET, verified before replacing the
        # redundant uncompressed representation. No experiment data is lost.
        SOURCE.unlink()
        print(json.dumps(record, indent=2))
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
