# Shared release procedure

Read this file from either release skill. The calling skill supplies the channel,
version, build version, tag, and notes path. Follow these checks in order; the
release script remains the source of truth for commands and supported arguments.

## Prepare the source and destination

- Read `Scripts/release.sh`, `Scripts/package-app.sh`, and
  `Scripts/verify-packaged-app.sh`. Inspect the working tree, selected commit,
  remotes, target GitHub repository, and public feed from `SUFeedURL`.
  Confirm the push remote and release repository refer to the same project.
- List the proposed changes and any unrelated tracked or untracked work. Resolve
  release scope with the user before committing or packaging mixed work. Preserve
  unrelated changes; do not automatically stash, discard, or include them.
- Check that public documentation and available `.vscode/docs/` still match the
  release behavior. Prepare the calling skill's notes, commit only approved
  changes, and record the source commit. Require a clean checkout, including
  untracked files, before running the release script.
- Save the current latest stable release identity and signed public feed for
  comparison after publication. An authentication, network, or verification
  failure is not evidence that a version or tag is unused.

Done when the source scope and destination are known and the notes match the
selected commit.

## Verify locally and check credentials

- Complete [build-local](build-local/SKILL.md) for the selected source and version.
  Pass explicit `APP_VERSION` and `BUILD_VERSION` to local packaging. Keep
  `NOTARY_PROFILE` unset for this local-only stage, restoring it only for release.
  If changing source after validation, rerun affected tests and packaging checks.
- Require an available Developer ID signing identity, authenticated `gh` for the
  target repository, the Sparkle signing tools, and a usable notary Keychain
  profile. Respect the current user's credential instructions. Use an explicitly
  configured `NOTARY_PROFILE`; if none is supplied, ask rather than guess from
  an old validation date. Check the selected profile with
  `xcrun notarytool history --keychain-profile "$NOTARY_PROFILE"`. If unavailable,
  stop; do not retrieve, print, or log credentials or switch profiles silently.
  The Sparkle signing account and the notary profile are separate settings.
- Confirm the script builds with `WHALLM_BUILD_FLAVOR=distribution` and verifies
  the signed, notarized App and ZIP before publication. Verification must cover
  signatures, notarization, English/Simplified Chinese/Traditional Chinese
  resources, and startup without the project `.build` directory.

Done when local verification passes and release signing prerequisites are usable.

## Approve and publish

- Present repository, commit, channel, App/build versions, tag, notes, and the
  intended tag push, GitHub Release, and Pages update. Obtain approval of this
  exact publication plan before creating/pushing the tag or publishing.
- Create the agreed tag at the verified commit and push only that tag. If a tag
  already exists locally or remotely, verify its peeled commit; never overwrite
  it. Stop on mismatch. An existing GitHub Release requires recovery, not another
  `gh release create`.
- Verify `HEAD`, the local tag, and the remote tag all resolve to the same commit
  and the checkout remains clean. The release script requires these conditions
  before packaging and rechecks them before upload.
- Run the calling skill's `make release` command with explicit channel, version,
  tag, build version, and notes path. Preserve the script's pre-upload checks and
  bounded Pages wait. Stop on failure; no automatic retry loop or force push.

## Verify the published result

- Use the script's freshly downloaded GitHub ZIP and `appcast.xml`, not just local
  build files. Require downloaded archive verification to pass again, including
  notarization, localization, and isolated startup. Verify the feed signature,
  attachment identities, and checksums against the uploaded artifacts.
- Check the calling skill's GitHub prerelease/latest and update-channel rules.
  Verify the public feed actually serves the expected signed contents after the
  Pages deployment; a successful `gh-pages` push alone is insufficient.
- Report the source commit, versions, tag, release URL, SHA-256 of both downloaded
  attachments, and any checks that could not run. Keep validation records in the
  existing local operations documentation when available.

Complete only when artifacts and public feed pass. If upload succeeds but download
verification, channel checks, or Pages verification fails, report **partially
published**, with the exact completed stage. Preserve the existing tag and release;
inspect the remote state and obtain approval for a targeted recovery. Never delete,
replace, or recreate published artifacts merely to make a rerun pass.
