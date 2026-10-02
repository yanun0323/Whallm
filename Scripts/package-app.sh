#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h}
python_executable=${PYTHON_EXECUTABLE:-$project_root/.venv/bin/python}
output_root=${OUTPUT_ROOT:-$project_root/dist}
app_path=$output_root/Whallm.app
zip_path=$output_root/Whallm-macOS-arm64.zip
app_version=${APP_VERSION:-1.0.0}
build_version=${BUILD_VERSION:-$app_version}
build_flavor=${WHALLM_BUILD_FLAVOR:-distribution}
swift_build_arguments=(--package-path "$project_root" -c release)
# Swift 6.4's Clang link command forwards --sysroot but can omit -isysroot,
# recording the deployment target as the SDK and selecting legacy AppKit controls.
# Pass the selected SDK explicitly to Clang as well; keep the deployment target.
sdk_path=$(xcrun --sdk macosx --show-sdk-path)
sdk_version=$(xcrun --sdk macosx --show-sdk-version)
swift_build_arguments+=(--sdk "$sdk_path"
  -Xswiftc -Xclang-linker -Xswiftc -isysroot
  -Xswiftc -Xclang-linker -Xswiftc "$sdk_path")
case $build_flavor in
  local) swift_build_arguments+=(-Xswiftc -DWHALLM_LOCAL_BUILD) ;;
  distribution) ;;
  *) print -u2 "WHALLM_BUILD_FLAVOR must be local or distribution."; exit 1 ;;
esac

if [[ $app_version != <->(|.<->)(|.<->) ||
      ( $build_version != <->(|.<->)(|.<->) &&
        $build_version != <->(|.<->)(|.<->)d<1-255> ) ]]; then
  print -u2 "APP_VERSION must be numeric; BUILD_VERSION may also end in d1 through d255."
  exit 1
fi

if [[ ! -x $python_executable ]]; then
  print -u2 "Python environment not found: $python_executable"
  print -u2 "Create .venv and install requirements.txt before packaging."
  exit 1
fi

TRANSFORMERS_VERBOSITY=error \
  $python_executable -c 'import mlx, numpy, sentencepiece, tiktoken, transformers'

# Packaging copies the selected environment; changing requirements alone does
# not update it. Reject a stale MLX runtime before creating an affected App.
$python_executable - "$project_root/requirements.txt" <<'PY'
import importlib.metadata
from pathlib import Path
import sys

expected = next(line.split("==", 1)[1] for line in
                Path(sys.argv[1]).read_text().splitlines() if line.startswith("mlx=="))
for package in ("mlx", "mlx-metal"):
    actual = importlib.metadata.version(package)
    if actual != expected:
        raise SystemExit(f"{package}=={actual}; packaging requires {expected}. "
                         "Install requirements.txt in the selected Python environment first.")
PY

python_version=$($python_executable -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python_framework=$($python_executable -c 'import pathlib, sys; print(pathlib.Path(sys.base_prefix).parents[1])')
site_packages=$($python_executable -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
python_binary=$($python_executable -c 'import os, sys; print(os.path.realpath(sys.executable))')
ane_bridge=$project_root/.build/native/libWhallmANE.dylib

if [[ ! -d $python_framework || ! -d $site_packages ]]; then
  print -u2 "The selected Python environment cannot be bundled."
  exit 1
fi

# A standalone interpreter resolves to an unrelated directory that the check
# above accepts, so confirm the framework layout the rewrites below require.
if [[ ${python_framework:t} != Python.framework ||
      ! -f $python_framework/Versions/$python_version/Python ]]; then
  print -u2 "Not a framework build: $python_executable"
  print -u2 "Packaging bundles Python.framework, but the environment resolved to"
  print -u2 "$python_framework."
  print -u2 "Recreate the environment from a framework Python such as"
  print -u2 "/opt/homebrew/opt/python@$python_version/bin/python$python_version."
  exit 1
fi

swift build "${swift_build_arguments[@]}" --product dsv4-app
"$project_root/Scripts/build-ane-bridge.sh" "$ane_bridge"
binary_path=$(swift build "${swift_build_arguments[@]}" --show-bin-path)/dsv4-app
resource_bundle=${binary_path:h}/DeepSeekV4SSD_DeepSeekV4SSDApp.bundle
sparkle_framework=$project_root/.build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework

if [[ ! -d $resource_bundle ]]; then
  print -u2 "App resource bundle not found: $resource_bundle"
  exit 1
fi
if [[ ! -d $sparkle_framework ]]; then
  print -u2 "Sparkle framework not found: $sparkle_framework"
  exit 1
fi

rm -rf "$app_path" "$zip_path"
mkdir -p "$app_path/Contents/MacOS" "$app_path/Contents/Resources/python" "$app_path/Contents/Frameworks"
ditto "$binary_path" "$app_path/Contents/MacOS/dsv4-app"
ditto "$project_root/Packaging/Info.plist" "$app_path/Contents/Info.plist"
ditto "$project_root/Packaging/AppIcon.icns" "$app_path/Contents/Resources/AppIcon.icns"
ditto "$sparkle_framework" "$app_path/Contents/Frameworks/Sparkle.framework"
ditto "$ane_bridge" "$app_path/Contents/Frameworks/libWhallmANE.dylib"
ditto "$project_root/Native/ANEBridge/LICENSE" \
  "$app_path/Contents/Resources/ANEBridge-LICENSE"
ditto "$resource_bundle" "$app_path/Contents/Resources/${resource_bundle:t}"
ditto "$project_root/runtime" "$app_path/Contents/Resources/runtime"
model_packages=$project_root/Sources/DeepSeekRepack/Resources/ModelPackages.json
ditto "$model_packages" "$app_path/Contents/Resources/ModelPackages.json"
ditto "$model_packages" "$app_path/Contents/Resources/runtime/deepseek_v4_ssd/model_support/ModelPackages.json"
qwen_vision=$project_root/Sources/DeepSeekRepack/Resources/QwenVision.json
ditto "$qwen_vision" "$app_path/Contents/Resources/QwenVision.json"
ditto "$qwen_vision" "$app_path/Contents/Resources/runtime/deepseek_v4_ssd/qwen_vision/QwenVision.json"
PYTHONPATH="$app_path/Contents/Resources/runtime" \
  $python_executable -m deepseek_v4_ssd.model_support
for localization in "$project_root/Sources/DeepSeekV4SSDApp/Resources"/*.lproj; do
  ditto "$localization" "$app_path/Contents/Resources/${localization:t}"
done
ditto "$site_packages" "$app_path/Contents/Resources/python/site-packages"
ditto "$python_framework" "$app_path/Contents/Frameworks/Python.framework"
# Homebrew creates the framework symlinks only in a linked keg, and both
# codesign and the PYTHONHOME the App resolves need them, so restore any that
# the copied environment left out.
bundled_framework=$app_path/Contents/Frameworks/Python.framework
if [[ ! -e $bundled_framework/Versions/Current ]]; then
  ln -s "$python_version" "$bundled_framework/Versions/Current"
fi
for framework_link in Python Resources Headers; do
  if [[ ! -e $bundled_framework/$framework_link ]]; then
    ln -s "Versions/Current/$framework_link" "$bundled_framework/$framework_link"
  fi
done
if [[ -d ${python_framework:h}/Libraries ]]; then
  ditto "${python_framework:h}/Libraries" "$app_path/Contents/Frameworks/Libraries"
fi
ditto "$python_binary" "$app_path/Contents/MacOS/python3"
rm -f "$app_path/Contents/Frameworks/Python.framework/Versions/$python_version/lib/python$python_version/site-packages"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $app_version" "$app_path/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $build_version" "$app_path/Contents/Info.plist"
install_name_tool -add_rpath "@executable_path/../Frameworks" "$app_path/Contents/MacOS/dsv4-app"

rewrite_python_library() {
  local target=$1
  local replacement=$2
  local dependency
  if otool -L "$target" | /usr/bin/grep -Fq "$replacement"; then
    return
  fi
  dependency=$(otool -L "$target" | tail -n +2 | sed -n \
    '/Python\.framework\/Versions\//{s/^[[:space:]]*//;s/ (.*$//;p;q;}')
  if [[ -z $dependency ]]; then
    print -u2 "Python framework dependency not found: $target"
    exit 1
  fi
  install_name_tool -change "$dependency" "$replacement" "$target"
}

framework_version_path="$app_path/Contents/Frameworks/Python.framework/Versions/$python_version"
rewrite_python_library \
  "$app_path/Contents/MacOS/python3" \
  "@executable_path/../Frameworks/Python.framework/Versions/$python_version/Python"
rewrite_python_library \
  "$framework_version_path/bin/python$python_version" \
  "@executable_path/../Python"
rewrite_python_library \
  "$framework_version_path/Resources/Python.app/Contents/MacOS/Python" \
  "@executable_path/../../../../Python"
install_name_tool -id \
  "@rpath/Python.framework/Versions/$python_version/Python" \
  "$framework_version_path/Python"

library_path="$app_path/Contents/Frameworks/Libraries"
mkdir -p "$library_path"
while true; do
  copied_library=false
  while IFS= read -r -d '' target; do
    if ! /usr/bin/file -b "$target" | /usr/bin/grep -q 'Mach-O'; then
      continue
    fi
    while IFS= read -r dependency; do
      [[ $dependency == /opt/homebrew/* ]] || continue
      library_name=${dependency:t}
      bundled_library="$library_path/$library_name"
      if [[ ! -f $bundled_library ]]; then
        dependency_source=$($python_executable -c \
          'import os, sys; print(os.path.realpath(sys.argv[1]))' "$dependency")
        ditto "$dependency_source" "$bundled_library"
        copied_library=true
      fi
      loader_relative_path=$($python_executable -c \
        'import os, sys; print(os.path.relpath(sys.argv[1], os.path.dirname(sys.argv[2])))' \
        "$bundled_library" "$target")
      install_name_tool -change "$dependency" \
        "@loader_path/$loader_relative_path" "$target"
    done < <(otool -L "$target" | tail -n +2 | sed \
      's/^[[:space:]]*//;s/ (.*$//')
  done < <(/usr/bin/find "$app_path/Contents" -type f -print0)
  [[ $copied_library == true ]] || break
done
for bundled_library in "$library_path"/*.dylib(N); do
  install_name_tool -id "@rpath/${bundled_library:t}" "$bundled_library"
done

/usr/bin/find "$app_path" -type d -name __pycache__ -prune -exec rm -rf {} +
/usr/bin/find "$app_path" -type f -name '*.pyc' -delete
rm -rf "$app_path/Contents/Frameworks/Python.framework/Versions/$python_version/lib/python$python_version/test"

code_sign_identity=${CODE_SIGN_IDENTITY:--}
code_sign_arguments=(--force --sign "$code_sign_identity")
code_sign_metadata=identifier,entitlements
if [[ $code_sign_identity != - ]]; then
  code_sign_arguments+=(--options runtime --timestamp)
  code_sign_metadata+=,flags
fi
while IFS= read -r -d '' target; do
  if /usr/bin/file -b "$target" | /usr/bin/grep -q 'Mach-O'; then
    codesign "${code_sign_arguments[@]}" \
      --preserve-metadata="$code_sign_metadata" "$target"
  fi
done < <(/usr/bin/find "$app_path/Contents" -type f -print0)
while IFS= read -r -d '' target; do
  codesign "${code_sign_arguments[@]}" \
    --preserve-metadata="$code_sign_metadata" "$target"
done < <(
  /usr/bin/find "$app_path/Contents" -depth -type d \
    \( -name '*.app' -o -name '*.framework' -o -name '*.bundle' -o -name '*.xpc' \) \
    -print0
)
codesign "${code_sign_arguments[@]}" "$app_path"
codesign --verify --deep --strict "$app_path"

PYTHONHOME="$app_path/Contents/Frameworks/Python.framework/Versions/Current" \
PYTHONPATH="$app_path/Contents/Resources/runtime:$app_path/Contents/Resources/python/site-packages" \
PYTHONDONTWRITEBYTECODE=1 \
  "$app_path/Contents/MacOS/python3" -c 'import mlx, deepseek_v4_ssd.server'

ditto -c -k --sequesterRsrc --keepParent "$app_path" "$zip_path"

if [[ -n ${NOTARY_PROFILE:-} ]]; then
  if [[ $code_sign_identity == - ]]; then
    print -u2 "CODE_SIGN_IDENTITY is required when NOTARY_PROFILE is set."
    exit 1
  fi
  xcrun notarytool submit "$zip_path" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$app_path"
  rm -f "$zip_path"
  ditto -c -k --sequesterRsrc --keepParent "$app_path" "$zip_path"
fi

EXPECTED_LOCAL_BUILD=$([[ $build_flavor == local ]] && print 1 || print 0) \
EXPECTED_SDK_VERSION="$sdk_version" \
REQUIRE_NOTARIZATION=$([[ -n ${NOTARY_PROFILE:-} ]] && print 1 || print 0) \
  "$project_root/Scripts/verify-packaged-app.sh" "$app_path" "$zip_path"

print "App: $app_path"
print "Archive: $zip_path"
