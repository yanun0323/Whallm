# Project instructions

Use the ubiquitous language in `CONTEXT.md`.

## App packaging behaviors

Treat these as workflow requests and read the corresponding skill before starting:

- `build local`: [.agents/skills/build-local/SKILL.md](.agents/skills/build-local/SKILL.md).
- `release new version`: [.agents/skills/release-new-version/SKILL.md](.agents/skills/release-new-version/SKILL.md).

## Documentation

Public documentation must be readable from a fresh checkout. Keep `README*.md`,
`BENCHMARK.md`, and `CONTEXT.md` free of links to local documentation or archives.
`.vscode/` contains local files excluded from version control; keep them local.

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
