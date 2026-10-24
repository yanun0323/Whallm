---
name: build-local
description: Build and verify local Whallm App and ZIP packages. Use when the user requests `build local` or before publishing a stable or dev release.
---

# Build local

1. Run the applicable tests.
2. Run `make package` to create `dist/Whallm.app` and
   `dist/Whallm-macOS-arm64.zip`.
3. Verify the App with `codesign --verify --deep --strict`.
4. Extract the ZIP to a temporary directory. Verify the extracted App again.
5. Start the packaged App in an isolated check that denies access to the project
   `.build` directory. Confirm that the App stays open and does not trap during
   `L10n` initialization.
6. Confirm that the packaged App loads the English, Simplified Chinese, and
   Traditional Chinese localization files from `Contents/Resources` before it
   accesses `Bundle.module`.

The packaged App must not require an absolute path from the build machine.
Keep local packaging on the local machine. Do not create a tag, notarize an
artifact, or upload an artifact.

`build local` is complete only when the App and the extracted ZIP pass all
checks.
