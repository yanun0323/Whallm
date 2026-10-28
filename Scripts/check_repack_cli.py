"""Check repack dispatch offline: python Scripts/check_repack_cli.py CLI_PATH."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def main():
    executable = Path(sys.argv[1]).resolve()
    with TemporaryDirectory(prefix="whallm-repack-check-") as directory:
        root = Path(directory)
        # Intentionally incompatible plans must reach each family's validator,
        # which rejects them before reading checkpoint data or creating files.
        for kind, version, message in (
            (None, 1, "pinned model contract"),
            ("deepseek-v4", 1, "pinned model contract"),
            ("deepseek-v4.1", 3, "pinned V4.1 model contract"),
            ("deepseek-v4.1-1017", 3, "pinned V4.1 model contract"),
            ("qwen3.8-flash-next", 2, "pinned Qwen model contract"),
        ):
            plan = dict(
                formatVersion=version, modelID="invalid", revision="invalid",
                layerCount=0, expertCount=0, selectedExpertCount=0,
                expertBlobSize=0, checkpointTensorBytes=0, files=[],
                commonTensors=[], expertRegions=[], copies=[],
            )
            if kind is not None:
                plan["modelKind"] = kind
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            destination = root / "model.dsv4"
            result = subprocess.run(
                [str(executable), "repack", "--plan", str(plan_path),
                 "--output", str(destination)],
                capture_output=True, text=True, timeout=10,
            )
            assert result.returncode == 1 and message in result.stderr, result.stderr
            assert not destination.exists()
            print(f"PASS: {kind or 'legacy V4 plan'}")


if __name__ == "__main__":
    main()
