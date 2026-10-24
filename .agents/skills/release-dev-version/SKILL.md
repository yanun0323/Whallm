---
name: release-dev-version
description: Build, verify, and publish a Whallm dev prerelease. Use when the user requests `release dev version`.
---

# Release dev version

A dev release is a signed, notarized distribution build published to the dev
update channel, not a local build with fewer checks. For local-only packages,
use [build-local](../build-local/SKILL.md).

1. Read [the shared release procedure](../release-common.md). Complete its
   source and destination checks before choosing a version.
2. Inspect local and remote tags, GitHub Releases, and the signed public update
   feed. Use numeric version ordering, not tag dates or lexical sorting.
   If the user supplies a version or dev number, validate it against all three.
   Otherwise continue the single active dev series newer than the latest stable
   version; if none exists, increment the latest stable patch version. If several
   series are active, records disagree, or remote inspection fails, stop and ask.
3. For the selected `X.Y.Z`, choose one more than the highest reserved or
   published dev number across those records, starting at 1. Require `N` in
   `1..255`. Stop at exhaustion; never reuse a published number or move a tag.
   The identities must be:
   - App version: `X.Y.Z`
   - Build version: `X.Y.ZdN`
   - Git tag: `vX.Y.Z-dev.N`
4. Verify ordering with the bundled Sparkle version comparator, not a generic
   SemVer parser: the proposed build must be newer than the latest stable and
   preceding dev build, and older than the eventual `X.Y.Z` stable build.
   If the comparator cannot be exercised, stop rather than assume the order.
5. Write English notes in `Packaging/ReleaseNotes/X.Y.Z-dev.N.md`. Describe changes
   since the preceding dev release of this series, or the latest stable release
   for the first dev build. Identify known limitations and required upgrade steps;
   distinguish fixed-output tests from real-model validation. Keep stable notes
   unchanged unless independently inaccurate.
6. Complete the shared preparation, approval, and tag checks, then publish using
   the existing script (replace the placeholders):

   ```sh
   make release VERSION=X.Y.Z CHANNEL=dev DEV_BUILD=N \
     TAG=vX.Y.Z-dev.N BUILD_VERSION=X.Y.ZdN \
     RELEASE_NOTES_FILE=Packaging/ReleaseNotes/X.Y.Z-dev.N.md
   ```

7. Complete shared artifact and public-feed verification. Additionally verify
   that GitHub marks this release as a prerelease, the previous latest stable
   release remains latest, and the new feed item has `sparkle:channel` equal to
   `dev`. Compare against the saved pre-release feed: its stable item and download
   URL must remain available. Check stable/dev channel selection using the App's
   existing Sparkle integration; do not change the user's saved update channel.

Complete only when the downloaded artifacts and public dev feed pass verification.
Report the commit, App/build versions, tag, URL, SHA-256 values, and validation
limits. Follow the shared partial-publication rules if any later check fails.
