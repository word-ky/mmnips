# Project Brief

## Goal

Use the local machine for Claude Code and use AutoDL only as the remote execution host for training, inference, and artifact storage.

## Local Control Surface

- Project root: `D:\work\claude-autodl`
- Workflow scripts live under `scripts/`
- Connection and remote layout live in `.autodl/config.json`
- Analysis-pack files live in `context/`

## Remote Layout

- Remote base: `/autodl-fs/data/1401/claude-autodl`
- Immutable code snapshots: `releases/<releaseId>`
- Active code symlink: `current`
- Per-run outputs: `runs/<runId>`
- Persistent datasets and checkpoints: `shared/`

## Guardrails

- Do not install or run Claude Code on AutoDL.
- Do not place Anthropic credentials on AutoDL.
- Keep large datasets and checkpoints under `shared/`; do not redeploy them with code releases.
- Prefer the existing workflow scripts over ad-hoc SSH commands.

## Common Operations

- Deploy code: `pwsh scripts/autodl-deploy.ps1 -Tag <tag>`
- Run a remote job: `pwsh scripts/autodl-run.ps1 -Name <name> -Cmd "<command>"`
- Follow logs: `pwsh scripts/autodl-logs.ps1 -RunId <run-id> -Follow`
- Fetch results: `pwsh scripts/autodl-fetch.ps1 -RunId <run-id>`
- Export a web-analysis bundle: `pwsh scripts/export-analysis-pack.ps1`

## How To Use With Web GPT

Upload these files together when asking for a detailed analysis:

1. `context/project-brief.md`
2. `context/latest-status.md`
3. `context/latest-log-tail.md`
4. `context/decision-log.md`
5. `prompts/web-analysis.md`
