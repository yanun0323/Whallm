"""Write one authenticated Xet byte range to a temporary file for Swift repacking."""

from pathlib import Path
import sys
import time

import hf_xet
from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url


def write_unordered_range(stream, output, expected_size: int) -> int:
    """Assemble completed Xet chunks without waiting for earlier offsets."""
    received = 0
    spans = []
    last_reported = 0
    last_report_time = time.monotonic()
    for offset, chunk in stream:
        chunk_end = offset + len(chunk)
        if not chunk or offset < 0 or chunk_end > expected_size:
            raise ValueError("Xet returned an invalid range chunk")
        output.seek(offset)
        output.write(chunk)
        spans.append((offset, chunk_end))
        received += len(chunk)
        now = time.monotonic()
        if received - last_reported >= 1024 * 1024 and now - last_report_time >= 0.2:
            print(received, flush=True)
            last_reported = received
            last_report_time = now

    position = 0
    for start, end in sorted(spans):
        if start != position:
            raise ValueError("Xet returned overlapping or missing range chunks")
        position = end
    if position != expected_size or received != expected_size:
        raise ValueError("Xet returned an incomplete range")
    if received != last_reported:
        print(received, flush=True)
    return received


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
    try:
        with destination.open("xb") as output, group:
            info = hf_xet.XetFileInfo(metadata.xet_file_data.file_hash, file_size=metadata.size)
            write_unordered_range(
                group.download_unordered_stream(info, start=start, end=end), output, end - start)
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
