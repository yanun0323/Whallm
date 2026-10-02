#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h}
app_path=${1:?Usage: verify-packaged-app.sh APP_PATH ZIP_PATH}
zip_path=${2:?Usage: verify-packaged-app.sh APP_PATH ZIP_PATH}
require_notarization=${REQUIRE_NOTARIZATION:-0}
expected_local_build=${EXPECTED_LOCAL_BUILD:-}
expected_sdk_version=${EXPECTED_SDK_VERSION:-}
verification_root=$(mktemp -d)
trap 'rm -rf "$verification_root"' EXIT

verify_signature() {
  local target=$1
  codesign --verify --deep --strict "$target"
  if [[ -n $expected_sdk_version ]]; then
    local actual_sdk
    actual_sdk=$(otool -l "$target/Contents/MacOS/dsv4-app" | awk '
      $1 == "cmd" { build_version = ($2 == "LC_BUILD_VERSION") }
      build_version && $1 == "sdk" { print $2; exit }
    ')
    if [[ $actual_sdk != $expected_sdk_version ]]; then
      print -u2 "Packaged App SDK mismatch: expected $expected_sdk_version, found $actual_sdk. Native control appearance may regress."
      exit 1
    fi
    print "App SDK verified: $actual_sdk"
  fi
  if [[ $require_notarization == 1 ]]; then
    xcrun stapler validate "$target"
    spctl --assess --type execute --verbose=2 "$target"
  fi
}

verify_localizations() {
  local target=$1
  local usage_description
  local language
  usage_description=$(/usr/libexec/PlistBuddy \
    -c 'Print :NSLocalNetworkUsageDescription' "$target/Contents/Info.plist")
  [[ -n $usage_description ]] || {
    print -u2 "NSLocalNetworkUsageDescription is missing."
    exit 1
  }
  for language in en zh-Hans zh-Hant; do
    local localization=$target/Contents/Resources/$language.lproj
    local strings
    for strings in "$localization/Localizable.strings" "$localization/InfoPlist.strings"; do
      [[ -f $strings ]] || {
        print -u2 "Localization is missing: $strings"
        exit 1
      }
      plutil -lint "$strings" >/dev/null
    done
  done
}

verify_model_packages() {
  local target=$1
  local catalog=$target/Contents/Resources/ModelPackages.json
  local runtime_catalog=$target/Contents/Resources/runtime/deepseek_v4_ssd/model_support/ModelPackages.json
  [[ -f $catalog && -f $runtime_catalog ]] || {
    print -u2 "Model package catalog is missing."
    exit 1
  }
  cmp "$catalog" "$runtime_catalog"
  cmp "$target/Contents/Resources/QwenVision.json" \
    "$target/Contents/Resources/runtime/deepseek_v4_ssd/qwen_vision/QwenVision.json"
  local profile="(version 1)(allow default)"
  profile+="(deny file-read* (subpath \"$project_root/.build\"))"
  profile+="(deny file-read* (subpath \"$project_root/.venv\"))"
  profile+="(deny file-read* (subpath \"$project_root/Sources\"))"
  PYTHONHOME="$target/Contents/Frameworks/Python.framework/Versions/Current" \
  PYTHONPATH="$target/Contents/Resources/runtime:$target/Contents/Resources/python/site-packages" \
  PYTHONDONTWRITEBYTECODE=1 \
    sandbox-exec -p "$profile" "$target/Contents/MacOS/python3" \
      -m deepseek_v4_ssd.model_support
}

launch_without_module_bundle_access() {
  local target=$1
  local module_bundle=$target/Contents/Resources/DeepSeekV4SSD_DeepSeekV4SSDApp.bundle
  local build_path=$project_root/.build
  local sandbox_profile
  local language

  [[ -d $module_bundle ]] || {
    print -u2 "Swift resource bundle is missing: $module_bundle"
    exit 1
  }
  sandbox_profile="(version 1)(allow default)"
  sandbox_profile+="(deny file-read* (subpath \"$build_path\"))"
  sandbox_profile+="(deny file-read* (subpath \"$module_bundle\"))"

  for language in en zh-Hans zh-Hant; do
    local log_path=$verification_root/launch-$language.log
    sandbox-exec -p "$sandbox_profile" \
      "$target/Contents/MacOS/dsv4-app" \
      --verify-localizations -appLanguage "$language" >"$log_path" 2>&1 &
    local app_pid=$!
    sleep 3
    if ! kill -0 "$app_pid" 2>/dev/null; then
      local exit_status=0
      wait "$app_pid" || exit_status=$?
      print -u2 \
        "Packaged App stopped during $language localization startup (status $exit_status)."
      sed -n '1,160p' "$log_path" >&2
      exit 1
    fi
    if ! /usr/bin/grep -Fxq "WHALLM_LOCALIZATION_READY:$language" "$log_path"; then
      kill -TERM "$app_pid"
      wait "$app_pid" || true
      print -u2 "Packaged App did not finish $language L10n initialization."
      sed -n '1,160p' "$log_path" >&2
      exit 1
    fi
    if ! /usr/bin/grep -Eq '^WHALLM_MODEL_PACKAGES_READY:[1-9][0-9]*$' "$log_path"; then
      kill -TERM "$app_pid"
      wait "$app_pid" || true
      print -u2 "Packaged App did not load its model packages."
      sed -n '1,160p' "$log_path" >&2
      exit 1
    fi
    if ! /usr/bin/grep -Fxq 'WHALLM_VISION_PACKAGES_READY:2' "$log_path"; then
      kill -TERM "$app_pid"
      wait "$app_pid" || true
      print -u2 "Packaged App did not load its Qwen image packages."
      exit 1
    fi
    print "Localization and model resources initialized without Keychain: $language"
    if [[ -n $expected_local_build ]]; then
      if ! /usr/bin/grep -Fxq "WHALLM_LOCAL_BUILD:$expected_local_build" "$log_path"; then
        kill -TERM "$app_pid"
        wait "$app_pid" || true
        print -u2 "Packaged App has the wrong local-build feature configuration."
        exit 1
      fi
      print "Local-build features verified: $expected_local_build"
    fi
    kill -TERM "$app_pid"
    wait "$app_pid" || true
  done
}

[[ -d $app_path && -f $zip_path ]] || {
  print -u2 "Packaged App or ZIP is missing."
  exit 1
}

verify_signature "$app_path"
verify_localizations "$app_path"
verify_model_packages "$app_path"
launch_without_module_bundle_access "$app_path"

ditto -x -k "$zip_path" "$verification_root/extracted"
extracted_app=$verification_root/extracted/Whallm.app
[[ -d $extracted_app ]] || {
  print -u2 "The ZIP does not contain Whallm.app."
  exit 1
}
verify_signature "$extracted_app"
verify_localizations "$extracted_app"
verify_model_packages "$extracted_app"
launch_without_module_bundle_access "$extracted_app"

print "Packaged App verification passed: $app_path"
print "Extracted ZIP verification passed: $zip_path"
