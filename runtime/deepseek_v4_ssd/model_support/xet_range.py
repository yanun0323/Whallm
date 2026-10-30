"""Write one authenticated Xet byte range to a temporary file for Swift repacking."""

import os
from pathlib import Path
import sys
import time

import hf_xet
from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url


def main() -> int:
    model_id, revision, path, start_text, end_text, destination_text = sys.argv[1:]
    start = int(start_text)
    end = int(end_text)
    if start < 0 or end <= start or not path.endswith(".safetensors"):
        raise ValueError("invalid Xet range")
    token = get_token()
    if not token:
        raise ValueError("HF read token unavailable")

    metadata = get_hf_file_metadata(hf_hub_url(model_id, path, revision=revision), token=token)
    if metadata.xet_file_data is None or metadata.size is None or end > metadata.size:
        raise ValueError("Xet range is outside the checkpoint file")

    session = hf_xet.XetSession()
    group = session.new_download_stream_group(
        token_refresh_url=metadata.xet_file_data.refresh_route,
        token_refresh_headers={"Authorization": "Bearer " + token},
    )
    destination = Path(destination_text)
    written = 0
    last_reported = 0
    last_report_time = time.monotonic()
    try:
        with destination.open("xb") as output, group:
            info = hf_xet.XetFileInfo(metadata.xet_file_data.file_hash, file_size=metadata.size)
            for chunk in group.download_stream(info, start=start, end=end):
                output.write(chunk)
                written += len(chunk)
                if written > end - start:
                    raise ValueError("Xet returned excess bytes")
                now = time.monotonic()
                if written - last_reported >= 8 * 1024 * 1024 and now - last_report_time >= 0.2:
                    print(written, flush=True)
                    last_reported = written
                    last_report_time = now
        if written != end - start:
            raise ValueError("Xet returned an incomplete range")
        if written != last_reported:
            print(written, flush=True)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Signed URLs and credentials must never appear in App logs.
        print(type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
