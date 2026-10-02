import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from Scripts import wait_for_pages as pages


ROOT = Path(__file__).resolve().parents[2]
COMMIT = "1" * 40
FEED_URL = "https://example.github.io/App/appcast.xml"
FEED = b"<rss>signed release feed</rss>"
DIGEST = hashlib.sha256(FEED).hexdigest()


class WaitForPagesTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.output = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.output))
        self.enterContext(patch.object(pages.time, "monotonic", side_effect=lambda: self.now))
        self.sleep = self.enterContext(patch.object(pages.time, "sleep", side_effect=self.advance))
        self.run = self.enterContext(patch.object(pages.subprocess, "run"))
        self.builds = [{"commit": COMMIT, "status": "built"}]
        self.feeds = [FEED]
        self.run.side_effect = self.respond

    def advance(self, seconds):
        self.now += seconds

    def respond(self, command, **kwargs):
        values = self.builds if command[0] == "gh" else self.feeds
        value = values.pop(0) if len(values) > 1 else values[0]
        if isinstance(value, Exception):
            raise value
        if isinstance(value, dict):
            value = json.dumps(value).encode()
        return subprocess.CompletedProcess(command, 0, stdout=value, stderr=b"")

    def wait(self, timeout=20):
        return pages.wait_for_pages("owner/repo", COMMIT, FEED_URL, FEED, timeout)

    def test_success_requires_completed_commit_and_exact_public_feed(self):
        self.assertEqual(self.wait(), DIGEST)
        self.sleep.assert_not_called()
        self.assertIn("Pages update feed verified", self.output.getvalue())
        commands = [call.args[0] for call in self.run.call_args_list]
        self.assertEqual(commands[0], ["gh", "api", "repos/owner/repo/pages/builds/latest"])
        self.assertEqual(commands[1][-1], FEED_URL)
        self.assertNotIn("?", commands[1][-1])
        self.assertNotIn("raw.githubusercontent.com", str(commands))

    def test_queued_build_then_success(self):
        self.builds.insert(0, {"commit": COMMIT, "status": "building"})
        self.assertEqual(self.wait(), DIGEST)
        self.assertEqual(self.now, 15)
        self.assertEqual(self.run.call_count, 3)

    def test_wrong_commit_cannot_pass_even_with_matching_feed(self):
        self.builds = [{"commit": "2" * 40, "status": "built"}]
        with self.assertRaisesRegex(pages.PagesWaitError, "within 20s"):
            self.wait()
        self.assertEqual(self.now, 20)
        self.assertTrue(all(call.args[0][0] == "gh" for call in self.run.call_args_list))

    def test_completed_deployment_with_stale_feed_times_out(self):
        self.feeds = [b"old signed feed"]
        with self.assertRaisesRegex(pages.PagesWaitError, "does not match"):
            self.wait()
        self.assertEqual(self.now, 20)
        self.assertNotIn("Pages update feed verified", self.output.getvalue())

    def test_cdn_can_catch_up_within_the_same_deadline(self):
        self.feeds = [b"old signed feed", FEED]
        self.assertEqual(self.wait(), DIGEST)
        self.assertEqual(self.now, 15)

    def test_failed_matching_build_stops_immediately(self):
        self.builds = [{"commit": COMMIT, "status": "errored", "error": {"message": "Deploy failed"}}]
        with self.assertRaisesRegex(pages.PagesWaitError, "Deploy failed"):
            self.wait()
        self.assertEqual(self.run.call_count, 1)
        self.sleep.assert_not_called()

    def test_failed_older_build_does_not_fail_the_new_commit(self):
        self.builds.insert(0, {"commit": "2" * 40, "status": "errored"})
        self.assertEqual(self.wait(), DIGEST)

    def test_unknown_build_state_is_not_success(self):
        self.builds = [{"commit": COMMIT, "status": "waiting"}]
        with self.assertRaisesRegex(pages.PagesWaitError, "Pages state: waiting"):
            self.wait()

    def test_queued_time_is_included_and_no_retry_is_submitted(self):
        self.builds = [{"commit": COMMIT, "status": "building"}]
        with self.assertRaisesRegex(pages.PagesWaitError, "No deployment retry was submitted"):
            self.wait(timeout=5)
        self.assertEqual(self.now, 5)
        self.sleep.assert_called_once_with(5)
        self.assertEqual(self.run.call_count, 1)
        self.assertNotIn("POST", str(self.run.call_args_list))

    def test_transient_request_failure_can_recover(self):
        self.builds.insert(0, subprocess.CalledProcessError(1, ["gh", "api"]))
        self.assertEqual(self.wait(), DIGEST)
        self.assertIn("Could not verify Pages yet", self.output.getvalue())

    def test_malformed_json_can_recover(self):
        self.builds.insert(0, b"not json")
        self.assertEqual(self.wait(), DIGEST)

    def test_invalid_build_response_fails_clearly(self):
        self.builds = [b"[]"]
        with self.assertRaisesRegex(pages.PagesWaitError, "invalid Pages build response"):
            self.wait()

    def test_network_timeouts_do_not_extend_the_overall_deadline(self):
        def timeout(command, **kwargs):
            if command[0] == "gh":
                return self.respond(command, **kwargs)
            self.advance(kwargs["timeout"])
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        self.run.side_effect = timeout
        with self.assertRaisesRegex(pages.PagesWaitError, "within 5s"):
            self.wait(timeout=5)
        self.assertEqual(self.now, 5)
        self.assertEqual(self.run.call_args.kwargs["timeout"], 5)
        self.sleep.assert_not_called()

    def test_each_command_uses_only_the_remaining_budget(self):
        def slow_response(command, **kwargs):
            self.advance(4 if command[0] == "gh" else 2)
            return self.respond(command, **kwargs)
        self.run.side_effect = slow_response
        with self.assertRaises(pages.PagesWaitError):
            self.wait(timeout=5)
        self.assertEqual([call.kwargs["timeout"] for call in self.run.call_args_list], [5, 1])
        self.assertNotIn("Pages update feed verified", self.output.getvalue())

    def test_request_timeouts_are_capped(self):
        self.wait(timeout=900)
        self.assertTrue(all(call.kwargs["timeout"] == 30 for call in self.run.call_args_list))

    def test_invalid_arguments_do_not_start_network_requests(self):
        invalid = [
            ("owner", COMMIT, FEED_URL, FEED, 20),
            ("owner/repo", "short-sha", FEED_URL, FEED, 20),
            ("owner/repo", COMMIT, "http://example.com/feed", FEED, 20),
            ("owner/repo", COMMIT, FEED_URL, b"", 20),
            ("owner/repo", COMMIT, FEED_URL, FEED, 0),
            ("owner/repo", COMMIT, FEED_URL, FEED, -1),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                pages.wait_for_pages(*args)
        self.run.assert_not_called()

    def test_cli_default_timeout_is_fifteen_minutes(self):
        self.assertEqual(pages.DEFAULT_TIMEOUT_SECONDS, 900)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "appcast.xml"
            path.write_bytes(FEED)
            self.assertEqual(pages.main([
                "--repository", "owner/repo", "--commit", COMMIT,
                "--feed-url", FEED_URL, "--expected-feed", str(path),
            ]), 0)
        self.assertIn("Waiting up to 900s", self.output.getvalue())

    def test_cli_reports_missing_expected_feed_without_network(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            code = pages.main([
                "--repository", "owner/repo", "--commit", COMMIT,
                "--feed-url", FEED_URL, "--expected-feed", str(Path(directory) / "missing.xml"),
            ])
        self.assertEqual(code, 1)
        self.run.assert_not_called()


class ReleaseWorkflowPolicyTests(unittest.TestCase):
    def test_all_research_workflows_are_manual_only(self):
        workflows = list((ROOT / ".github/workflows").glob("*.yml"))
        self.assertEqual(len(workflows), 6)
        for path in workflows:
            with self.subTest(path=path.name):
                text = path.read_text()
                triggers = re.search(r"^on:\n(.*?)(?=^\S)", text, re.MULTILINE | re.DOTALL)
                self.assertIsNotNone(triggers)
                self.assertEqual(triggers.group(1).strip(), "workflow_dispatch:")

    def test_release_waits_after_feed_push_before_reporting_success(self):
        script = (ROOT / "Scripts/release.sh").read_text()
        self.assertLess(script.index('push origin HEAD:gh-pages'), script.index('Scripts/wait_for_pages.py'))
        self.assertLess(script.index('Scripts/wait_for_pages.py'), script.index('print "Release:'))
        self.assertIn('--expected-feed "$download_root/appcast.xml"', script)
        self.assertIn('--timeout "$pages_wait_seconds"', script)
        self.assertIn("Print :SUFeedURL", script)
        self.assertIn("Do not recreate the release", script)

    def test_invalid_wait_limit_is_rejected_before_release_side_effects(self):
        for value in ("0", "-1", "not-a-number"):
            with self.subTest(value=value):
                result = subprocess.run(
                    ["/bin/zsh", str(ROOT / "Scripts/release.sh")],
                    env={**os.environ, "VERSION": "1.1.10", "PAGES_WAIT_SECONDS": value,
                         "PYTHON": "/does-not-exist/test-python"},
                    capture_output=True, text=True, timeout=5,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("PAGES_WAIT_SECONDS must be a positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
