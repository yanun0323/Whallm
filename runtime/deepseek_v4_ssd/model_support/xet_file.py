"""Stage one complete checkpoint shard through the same Xet path as `hf download`."""

from pathlib import Path
import shutil
import sys

import hf_xet  # noqa: F401 - hf_hub_download selects its native Xet path when installed.
from huggingface_hub import get_token, hf_hub_download
from tqdm import tqdm


class TransferProgress(tqdm):
    """Forward native Xet transfer updates to the Swift parent over stdout."""

    def __init__(self, *args, **kwargs):
        self._last_reported = 0
        super().__init__(*args, **kwargs)
        self._is_transfer = "downloading bytes" in self.desc.lower()

    def display(self, *args, **kwargs):
        pass

    def update(self, n=1):
        result = super().update(n)
        if self._is_transfer and self.n > self._last_reported:
            print(self.n, flush=True)
            self._last_reported = self.n
        return result


def main() -> int:
    model_id, revision, path, destination_text = sys.argv[1:]
    if not path.endswith(".safetensors"):
        raise ValueError("invalid Xet shard")
    token = get_token()
    if not token:
        raise ValueError("HF read token unavailable")

    destination = Path(destination_text)
    download_directory = destination.with_name(destination.name + ".download")
    try:
        result = hf_hub_download(
            repo_id=model_id,
            filename=path,
            revision=revision,
            token=token,
            local_dir=download_directory,
            tqdm_class=TransferProgress,
        )
        Path(result).replace(destination)
        print(destination.stat().st_size, flush=True)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(download_directory, ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Signed URLs and credentials must never appear in App logs.
        print(type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
