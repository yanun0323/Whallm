# Project instructions

Use the ubiquitous language in `CONTEXT.md`.

## App packaging behaviors

Treat these as workflow requests and read the corresponding skill before starting:

- `build local`: [.agents/skills/build-local/SKILL.md](.agents/skills/build-local/SKILL.md).
- `release new version`: [.agents/skills/release-new-version/SKILL.md](.agents/skills/release-new-version/SKILL.md).
- `release dev version`: [.agents/skills/release-dev-version/SKILL.md](.agents/skills/release-dev-version/SKILL.md).

## Documentation

Public documentation must be readable from a fresh checkout. Keep `README*.md`,
`BENCHMARK.md`, and `CONTEXT.md` free of links to local documentation or archives.
`.vscode/` contains local files excluded from version control; keep them local.

### README writing

Keep `README*.md` an entry point: what Whallm does, whether it fits the
reader's Mac, and how to start. Aim for roughly 100–140 lines per language;
clarity and essential warnings matter more than the count.

- Use plain language, short paragraphs and compact comparison tables. Explain
  necessary technical terms on first use; omit implementation details from the intro.
- Order content: introduction, requirements and models, quick start, performance
  summary, Codex connection, limits and privacy, documentation links, license.
- Keep detailed model settings and API contracts in `docs/usage.md`, full and
  historical measurements in `BENCHMARK.md`, and changes and upgrade steps in
  `Packaging/ReleaseNotes/`. Link rather than duplicate; prefer links over large
  collapsed sections. Keep test counts and packaging checklists out of READMEs.
- Keep only a small performance table in READMEs, with model, version, workload,
  units, source and evidence limits. Preserve measurements and caveats when moving
  them; incomplete observations must not become controlled comparison claims.
- Keep hardware and storage needs, preview status, concurrency limits and privacy
  warnings visible. Shortening must not hide requirements or unsupported features.
- Synchronize all five language versions unless the user limits the scope. Preserve
  language navigation, corresponding sections, facts and example behavior. Shared
  detailed references may stay in English; label them in translated READMEs.
- Before finishing, check every changed README's local links, images, anchors and
  code fences; compare examples and numbers across languages and their sources.
  Confirm moved content remains reachable and remove duplicated explanations.

### `.vscode/docs/`

When maintaining runtime behavior or recording validation, start with
`.vscode/docs/README.md` if it is available. It is the local documentation index
for the current project. If local documents are absent, use the code, tests,
and public documentation to establish current behavior.

Keep `.vscode/docs/` consistent with the current code, tests, installed model
contract, and reproducible measurements.
Update the applicable document when an API, default value, data path, limit, or
measured result changes.

Existing `.vscode/docs/` archives and manifests preserve historical benchmark
artifacts. Keep raw runs in `scratch/` and validated machine-readable artifacts
in `.vscode/docs/benchmarks/`; preserve historical archives as snapshots.
Record the commit, environment, workload, configuration, cache state, and output
token hash for each formal performance result.

Separate current validation, historical measurements, external facts, and
research hypotheses.
Do not write an estimate or an external benchmark as a project result.

### `.vscode/research/`

When continuing research, read `.vscode/research/README.md` and `.vscode/TODO.md`
if available. They contain research status and optional next steps.
`.vscode/research/` contains active investigation, experiment plans, source audits,
and unconfirmed technical directions.
Research files are not the source of truth for current runtime behavior.

Use primary sources for external technical claims.
State assumptions, test conditions, stop criteria, and evidence limits.
When the project adopts a research conclusion, update the applicable file in
`.vscode/docs/`.

The archives in `.vscode/research/` contain superseded plans and historical research.
Keep archived files for traceability.
Add a warning when a file no longer describes the current runtime.
Do not use an archived value as a current default or current performance result.
