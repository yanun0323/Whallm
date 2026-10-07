#!/usr/bin/env python3
"""Wait for a Pages commit and its exact signed appcast to be served publicly."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit


DEFAULT_TIMEOUT_SECONDS = 900
POLL_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 30


class PagesWaitError(RuntimeError):
    pass


def _run(command, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("The Pages wait limit has been reached.")
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        timeout=min(REQUEST_TIMEOUT_SECONDS, remaining),
    )
    return result.stdout


def wait_for_pages(repository, commit, feed_url, expected_feed, timeout_seconds=DEFAULT_TIMEOUT_SECONDS):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must be in owner/name form.")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use the full gh-pages commit SHA.")
    if urlsplit(feed_url).scheme != "https" or not urlsplit(feed_url).netloc:
        raise ValueError("The public update feed must use HTTPS.")
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError("The Pages wait limit must be a positive number of seconds.")
    if not expected_feed:
        raise ValueError("The expected signed update feed is empty.")

    expected_hash = hashlib.sha256(expected_feed).hexdigest()
    deadline = time.monotonic() + timeout_seconds
    last_message = None
    last_detail = "No matching Pages build has been observed."
    print(f"Waiting up to {timeout_seconds}s for Pages commit {commit} and {feed_url}", flush=True)

    while time.monotonic() < deadline:
        try:
            build = json.loads(_run([
                "gh", "api", f"repos/{repository}/pages/builds/latest",
            ], deadline))
            if not isinstance(build, dict):
                raise PagesWaitError("GitHub returned an invalid Pages build response.")
            if build.get("commit") != commit:
                last_detail = "The latest Pages build does not match the published commit yet."
            elif build.get("status") == "errored":
                message = (build.get("error") or {}).get("message") or "See the Pages workflow logs."
                raise PagesWaitError(f"Pages deployment failed: {message}")
            elif build.get("status") != "built":
                last_detail = f"Pages state: {build.get('status', 'unknown')}."
            else:
                # Use the same URL as the App, without a cache-busting query or
                # a raw.githubusercontent.com fallback that could hide a stale site.
                actual_feed = _run([
                    "curl", "--fail", "--silent", "--show-error", "--location",
                    "--proto", "=https", "--proto-redir", "=https",
                    "--max-time", str(REQUEST_TIMEOUT_SECONDS),
                    "--max-filesize", str(max(len(expected_feed), 1024 * 1024)),
                    feed_url,
                ], deadline)
                if time.monotonic() < deadline and hashlib.sha256(actual_feed).hexdigest() == expected_hash:
                    print(f"Pages update feed verified: SHA-256 {expected_hash}", flush=True)
                    return expected_hash
                last_detail = "Pages completed, but the public update feed does not match the signed release yet."
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            last_detail = f"Could not verify Pages yet ({type(error).__name__})."
        except TimeoutError:
            break

        if last_detail != last_message:
            print(last_detail, flush=True)
            last_message = last_detail
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(POLL_SECONDS, remaining))

    raise PagesWaitError(
        f"Pages was not verified within {timeout_seconds}s. {last_detail} "
        "No deployment retry was submitted. Check the Pages workflow before retrying."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--feed-url", required=True)
    parser.add_argument("--expected-feed", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        wait_for_pages(
            args.repository, args.commit, args.feed_url, args.expected_feed.read_bytes(), args.timeout,
        )
    except (PagesWaitError, ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
