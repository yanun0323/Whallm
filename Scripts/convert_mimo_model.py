"""Convert a complete local HF snapshot into a losslessly preserved Whallm artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepseek_v4_ssd.mimo.install import MODEL_ID, REVISION, convert, verify_reconstruction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        plan = json.loads((args.output / "checkpoint-map.json").read_text())
        print(json.dumps(verify_reconstruction(args.output, plan), indent=2))
        return
    if args.source is None:
        parser.error("--source must point to the complete downloaded snapshot")
    from huggingface_hub import HfApi
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("Hub revision mismatch")
    source_info = {"repository": MODEL_ID, "revision": REVISION,
                   "files": {f.rfilename: {"size": f.size, "sha256": f.lfs.sha256 if f.lfs else None,
                                           "blobID": f.blob_id} for f in info.siblings}}
    manifest = convert(args.source.expanduser().resolve(), args.output, source_info)
    print(json.dumps({"files": len(manifest["files"]),
                      "bytes": sum(f["size"] for f in manifest["files"]),
                      "revision": manifest["revision"]}, indent=2))


if __name__ == "__main__":
    main()
