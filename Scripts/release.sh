#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h}
version=${VERSION:?Set VERSION, for example: make release VERSION=1.1.0}
channel=${CHANNEL:-stable}
channel_arguments=()
release_arguments=()
case $channel in
  stable)
    tag=${TAG:-v$version}
    build_version=${BUILD_VERSION:-$version}
    ;;
  dev)
    dev_build=${DEV_BUILD:-1}
    [[ $dev_build == <1-255> ]] || {
      print -u2 "DEV_BUILD must be between 1 and 255."
      exit 1
    }
    tag=${TAG:-v$version-dev.$dev_build}
    build_version=${version}d${dev_build}
    [[ -z ${BUILD_VERSION:-} || $BUILD_VERSION == $build_version ]] || {
      print -u2 "Dev BUILD_VERSION must be $build_version; use DEV_BUILD to select the build."
      exit 1
    }
    channel_arguments=(--channel dev)
    release_arguments=(--prerelease --latest=false)
    ;;
  *)
    print -u2 "CHANNEL must be stable or dev."
    exit 1
    ;;
esac
repository=${GITHUB_REPOSITORY:-yanun0323/Whallm}
python_executable=${PYTHON:-$project_root/.venv/bin/python}
pages_wait_seconds=${PAGES_WAIT_SECONDS:-900}
[[ $pages_wait_seconds == <-> && $pages_wait_seconds -gt 0 ]] || {
  print -u2 "PAGES_WAIT_SECONDS must be a positive number of seconds."
  exit 1
}
command -v "$python_executable" >/dev/null || {
  print -u2 "Python is required to verify the public Pages update feed: $python_executable"
  exit 1
}
archive_name=Whallm-macOS-arm64.zip
archive_path=$project_root/dist/$archive_name
sparkle_tools=$project_root/.build/artifacts/sparkle/Sparkle/bin
notes_path=${RELEASE_NOTES_FILE:-$project_root/Packaging/ReleaseNotes/$version.md}
release_root=$(mktemp -d)
download_root=$(mktemp -d)
pages_root=$(mktemp -d)
trap 'rm -rf "$release_root" "$download_root" "$pages_root"' EXIT

if ! command -v gh >/dev/null || ! gh auth status >/dev/null 2>&1; then
  print -u2 "GitHub CLI is not signed in."
  exit 1
fi

if [[ -z ${CODE_SIGN_IDENTITY:-} || -z ${NOTARY_PROFILE:-} ]]; then
  print -u2 "CODE_SIGN_IDENTITY and NOTARY_PROFILE are required for a release."
  exit 1
fi

if [[ ! -f $notes_path ]]; then
  print -u2 "Release notes not found: $notes_path"
  exit 1
fi

if gh release view "$tag" --repo "$repository" >/dev/null 2>&1; then
  print -u2 "GitHub Release already exists: $tag"
  exit 1
fi

release_commit=$(git -C "$project_root" rev-parse HEAD)
verify_release_source() {
  [[ -z $(git -C "$project_root" status --porcelain) &&
     $(git -C "$project_root" rev-parse HEAD) == $release_commit &&
     $(git -C "$project_root" rev-parse "refs/tags/$tag^{commit}") == $release_commit &&
     $(gh api "repos/$repository/commits/$tag" --jq .sha) == $release_commit ]] || {
    print -u2 "Release requires a clean checkout and a local/remote tag matching the packaged commit."
    exit 1
  }
}
verify_release_source

# Keep this checkout until publication so a concurrent feed update rejects the push.
gh repo clone "$repository" "$pages_root" -- --branch gh-pages --single-branch --depth 1
"$sparkle_tools/sign_update" --account deepseek_ssd --verify "$pages_root/appcast.xml"
ditto "$pages_root/appcast.xml" "$release_root/appcast.xml"

WHALLM_BUILD_FLAVOR=distribution APP_VERSION=$version BUILD_VERSION=$build_version \
  "$project_root/Scripts/package-app.sh"

EXPECTED_LOCAL_BUILD=0 REQUIRE_NOTARIZATION=1 "$project_root/Scripts/verify-packaged-app.sh" \
  "$project_root/dist/Whallm.app" "$archive_path"

ditto "$archive_path" "$release_root/$archive_name"
ditto "$notes_path" "$release_root/${archive_name:r}.md"

"$sparkle_tools/generate_appcast" \
  --account deepseek_ssd \
  --download-url-prefix "https://github.com/$repository/releases/download/$tag/" \
  --link "https://github.com/$repository/releases/tag/$tag" \
  --embed-release-notes \
  --maximum-versions 1 \
  --versions "$build_version" \
  "${channel_arguments[@]}" \
  "$release_root"

"$sparkle_tools/sign_update" --account deepseek_ssd --verify "$release_root/appcast.xml"

verify_release_source
gh release create "$tag" \
  "$release_root/$archive_name" \
  "$release_root/appcast.xml" \
  --repo "$repository" \
  --title "${RELEASE_TITLE:-Whallm ${tag#v}}" \
  --verify-tag \
  --notes-file "$notes_path" \
  "${release_arguments[@]}"

gh release download "$tag" \
  --repo "$repository" \
  --pattern "$archive_name" \
  --pattern appcast.xml \
  --dir "$download_root"

[[ -f $download_root/$archive_name && -f $download_root/appcast.xml ]] || {
  print -u2 "GitHub Release does not contain the ZIP and appcast.xml."
  exit 1
}
EXPECTED_LOCAL_BUILD=0 REQUIRE_NOTARIZATION=1 "$project_root/Scripts/verify-packaged-app.sh" \
  "$project_root/dist/Whallm.app" "$download_root/$archive_name"

# Publish only after the downloaded release archive passes all checks.
ditto "$download_root/appcast.xml" "$pages_root/appcast.xml"
"$sparkle_tools/sign_update" --account deepseek_ssd --verify "$pages_root/appcast.xml"
git -C "$pages_root" add appcast.xml
git -C "$pages_root" commit -m "Publish $channel update $tag"
git -C "$pages_root" push origin HEAD:gh-pages

# A pushed commit is not proof that the App can see the update. Include queued
# jobs, network requests and stale CDN responses in one bounded wait.
pages_commit=$(git -C "$pages_root" rev-parse HEAD)
feed_url=$(/usr/libexec/PlistBuddy -c 'Print :SUFeedURL' \
  "$project_root/dist/Whallm.app/Contents/Info.plist")
if ! "$python_executable" "$project_root/Scripts/wait_for_pages.py" \
  --repository "$repository" \
  --commit "$pages_commit" \
  --feed-url "$feed_url" \
  --expected-feed "$download_root/appcast.xml" \
  --timeout "$pages_wait_seconds"; then
  print -u2 "Release $tag is already published, but its Pages update feed is not verified."
  print -u2 "Do not recreate the release. Inspect the Pages deployment and recheck the public feed."
  exit 1
fi

release_url=$(gh release view "$tag" --repo "$repository" --json url --jq .url)
print "Release: $release_url"
shasum -a 256 "$download_root/$archive_name" "$download_root/appcast.xml"
