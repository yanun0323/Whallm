---
name: release-new-version
description: Build, verify, and publish a Whallm release. Use when the user requests `release new version`.
---

# Release new version

1. Read and complete [build-local](../build-local/SKILL.md) first.
2. If the user does not give a version, read the latest stable `vX.Y.Z` Git tag
   and increment `Z` by one.
3. Confirm that the release changes, tests, public documentation, and available
   local documentation in `.vscode/docs/` are current.
   Write or update `Packaging/ReleaseNotes/X.Y.Z.md` in English for the GitHub
   Release, using the changes since the previous release. Summarize user-facing
   improvements, fixes, and any breaking changes or required upgrade steps.
   Verify that the notes match the actual changes and that the release script
   uses this file as its GitHub Release notes before publishing.
4. Require `CODE_SIGN_IDENTITY`, `NOTARY_PROFILE`, authenticated `gh`, and the
   Sparkle signing tools.
   On this Mac, use `NOTARY_PROFILE=deepseek_ssd` (verified 2026-09-07).
   Check it with `xcrun notarytool history --keychain-profile deepseek_ssd`
   before asking the user for a profile name; credentials remain in Keychain.
5. Before `gh release create`, require the release script to verify the final
   signed and notarized App, its ZIP, its localization files, and App startup
   without access to the project `.build` directory. Stop the release if a check
   fails.
6. Run `make release VERSION=X.Y.Z`.
7. After publication, download the GitHub Release ZIP to a temporary directory.
   Verify its signature, notarization ticket, localization files, and isolated
   startup again.
8. Confirm that the GitHub Release contains the ZIP and `appcast.xml`. Report the
   tag, release URL, and SHA-256 of both files.

`release new version` is complete only when the downloaded release artifacts
pass all checks.
