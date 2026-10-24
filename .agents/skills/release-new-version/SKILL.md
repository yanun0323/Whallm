---
name: release-new-version
description: Build, verify, and publish a Whallm release. Use when the user requests `release new version`.
---

# Release new version

1. Read [the shared release procedure](../release-common.md). Complete its source
   and destination checks before choosing a version.
2. If the user does not give a version, inspect local and remote stable `vX.Y.Z`
   tags, GitHub Releases, and the signed public update feed; increment the latest
   stable patch version. Stop on conflicting records or if an active dev series
   makes the intended stable version unclear. Validate any supplied version and
   confirm that its tag and release are not already published.
3. Write or update `Packaging/ReleaseNotes/X.Y.Z.md` in English using changes
   since the previous stable release, including changes already shipped in dev
   builds. Summarize user-facing improvements, fixes, known limitations, breaking
   changes, and required upgrade steps.
4. Complete shared local verification, credential, approval, and tag checks, then
   publish with explicit stable settings:

   ```sh
   make release VERSION=X.Y.Z CHANNEL=stable \
     TAG=vX.Y.Z BUILD_VERSION=X.Y.Z \
     RELEASE_NOTES_FILE=Packaging/ReleaseNotes/X.Y.Z.md
   ```

5. Complete shared artifact and public-feed verification. Verify the GitHub
   Release is not a prerelease and is the intended latest stable release. Check
   that its feed item is available to the stable channel and that Sparkle ranks
   it above preceding dev builds for the same version.

Complete only when downloaded release artifacts and the public stable feed pass
all checks. Report the shared evidence and follow its partial-publication rules
if any later check fails.
