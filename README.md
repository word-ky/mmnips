# Claude AutoDL Workflow

This project keeps `Claude Code` on the local machine and uses AutoDL only as a remote execution host.

## Layout

- `.autodl/config.json`: connection and remote layout
- `scripts/autodl-bootstrap.ps1`: one-time SSH key bootstrap and remote directory creation
- `scripts/autodl-deploy.ps1`: package local files and publish a new remote release
- `scripts/autodl-run.ps1`: start a remote job in `tmux` or `nohup`
- `scripts/autodl-logs.ps1`: inspect or follow job logs
- `scripts/autodl-fetch.ps1`: copy remote run outputs back to this machine
- `scripts/autodl-shell.ps1`: open an interactive remote shell
- `scripts/autodl-forward.ps1`: local port forwarding for TensorBoard or Jupyter
- `scripts/autodl-status.ps1`: quick remote health check
- `scripts/export-analysis-pack.ps1`: build a web-GPT-friendly status bundle in `context/`
- `scripts/publish-github.ps1`: optionally commit and push the analysis bundle to an existing Git repo

## Remote layout

All remote state lives under `/autodl-fs/data/1401/claude-autodl`:

- `releases/<releaseId>`: immutable deployed code snapshots
- `current`: symlink to the active release
- `runs/<runId>`: per-run logs, metadata, and artifacts
- `shared/`: datasets, checkpoints, caches, and anything too large to redeploy

## First-time setup

Run the bootstrap once with the current AutoDL password:

```powershell
pwsh scripts/autodl-bootstrap.ps1 -Password '<AUTODL_PASSWORD>'
```

This generates `.autodl/id_ed25519`, installs the public key into the remote `authorized_keys`, and creates the remote directory tree.

## Daily workflow

Deploy the current local files:

```powershell
pwsh scripts/autodl-deploy.ps1 -Tag train-fix
```

Run a job against the active release:

```powershell
pwsh scripts/autodl-run.ps1 -Name exp1 -Cmd "python train.py --config configs/exp1.yaml"
```

This repository also includes a smoke-test `train.py`, so you can verify the whole loop immediately:

```powershell
pwsh scripts/autodl-run.ps1 -Name smoke -Cmd "python train.py --steps 3 --message smoke-test"
```

Inspect the latest log:

```powershell
pwsh scripts/autodl-logs.ps1 -Follow
```

Fetch outputs back to `remote-runs/`:

```powershell
pwsh scripts/autodl-fetch.ps1
```

## Notes

- Put large datasets and checkpoints under `shared/`; do not redeploy them.
- `autodl-run.ps1` prefers `tmux` and falls back to `nohup` if `tmux` is unavailable.
- `autodl-forward.ps1` can expose remote services such as TensorBoard:

```powershell
pwsh scripts/autodl-forward.ps1 -LocalPort 6006 -RemotePort 6006
```

## Analysis Pack

Use the analysis-pack export when you want to hand the latest project state to web GPT or another reviewer without copy-pasting logs:

```powershell
pwsh scripts/export-analysis-pack.ps1 -LogLines 200
```

This refreshes:

- `context/latest-status.md`
- `context/latest-log-tail.md`
- `context/decision-log.md` (created if missing)
- `prompts/web-analysis.md`

If the project is already in a Git repo and you want those files on GitHub:

```powershell
pwsh scripts/publish-github.ps1 -Message "Update AutoDL analysis pack"
```

## Python Utilities

Small local Python helpers live under `tools/`:

- `python -m tools.list_context_files`: list tracked files in `context/` and `prompts/`
- `python -m tools.print_last_release`: print `.autodl/last-release`
- `python -m tools.list_fetched_runs`: show fetched runs under `remote-runs/`
- `python -m tools.summarize_fetched_run`: summarize the newest fetched run or a specific run id
