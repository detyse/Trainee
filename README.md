# Trainee

<img width="1254" height="1254" alt="ChatGPT Image Jun 25, 2026, 01_44_14 PM" src="https://github.com/user-attachments/assets/7bd146db-897b-4f2b-bf64-967e1f3324d1" />

Trainee is a conservative agent runtime for automating external model-training loops.

It does not contain your training code. Instead, it connects to an existing training project, runs the project’s own training command, watches for progress signals, extracts metrics, asks an LLM or a fallback decision policy for the next parameter set, and repeats until it stops.

The intended loop is:

```text
read project context -> run training -> monitor heartbeat -> parse metrics
-> compare with baseline / best-so-far -> decide next params -> continue or stop
```

## What Trainee provides

- Project initialization through `.trainee/project.yaml`.
- Automatic discovery of likely entrypoints, data directories, config files, environment type, and fixed training-limit flags.
- Structured launch commands for `system`, `uv`, `.venv`, and `conda` environments.
- A guarded execution mode using `bubblewrap`, where the project is read-only and only `.trainee/` is writable.
- Heartbeat monitoring from stdout, stderr, log-file modification times, or heartbeat JSON files.
- Metric extraction from stdout regexes, log regexes, JSONL files, and W&B summary files.
- Baseline-first research state, best-so-far tracking, hypothesis/change-summary tracking, and rejected-change avoidance.
- Session reports and ledgers exported as Markdown, CSV, JSONL, and JSON.
- A local Web UI and HTTP tool API for project setup, loop control, prompt preview, run inspection, provider settings, and reports.

## Requirements

- Python 3.9 or newer.
- Linux is recommended for guarded mode.
- `bubblewrap` / `bwrap` is required for guarded runs.
- `uv` is recommended for installation, but Trainee itself is a normal Python package.
- A working external training project with a command that can run from the terminal.

If `bwrap` is unavailable or the training job must write outside `.trainee/`, run explicitly with `--unsafe`.

## Installation

From this repository:

```bash
uv tool install --editable . --force
uv tool update-shell
```

Restart the shell if needed, then verify:

```bash
trainee version
trainee --help
```

For local development without installing globally:

```bash
uv run trainee --help
```

## Quick start

Run these commands inside the training project you want Trainee to control:

```bash
cd /path/to/training-project
trainee init
```

Then edit the generated config:

```bash
$EDITOR .trainee/project.yaml
```

Validate the project and inspect the final baseline command:

```bash
trainee doctor
trainee run --dry-run
```

Start the loop:

```bash
trainee run
```

Start the Web UI:

```bash
trainee webui
```

Open `http://127.0.0.1:8000` if the browser does not open automatically.

## Project initialization

`trainee init` creates:

- `.trainee/project.yaml` — the main project run configuration.
- `.trainee/context.md` — Trainee’s generated understanding of the project.
- `.trainee/README.md` — notes about detected candidates and local Trainee files.
- `.trainee/logs/`, `.trainee/runs/`, `.trainee/artifacts/` — runtime output locations.

Initialization is non-destructive by default. Existing files are kept.

```bash
trainee init
trainee init --baseline-config configs/base.yaml
trainee init --force
```

`--baseline-config` must point to an existing file inside the project. Trainee records it as `launch.baseline_config` and passes it to the launcher as `--config <absolute-path>`.

Detected config files are suggestions only. Trainee does not automatically choose `config.yaml`, `environment.yml`, or any other config as the baseline.

## Configuration: `.trainee/project.yaml`

The config file is the source of truth for CLI, Web UI, and tool API runs.

Example:

```yaml
version: 1

data:
  - path: data
    flag: --data-root

launch:
  environment: conda
  env_name: trainer
  command:
    - python
    - train.py
  baseline_config: configs/base.yaml
  args:
    - flag: --seed
      value: 7

run:
  max_rounds: 3
  timeout_minutes: 60
  fixed_args:
    - flag: --max-iter
      value: 1000

tuning:
  params:
    - name: lr
      flag: --lr
      type: float
      default: 0.001
      min_value: 0.00001
      max_value: 0.01

metrics:
  specs:
    - name: val_loss
      source: stdout_regex
      key_or_pattern: 'val_loss=(?P<value>-?\d+(?:\.\d+)?)'
      goal: min
      required: true
  prompt: "Use val_loss as the primary model-selection metric."

advanced:
  security_mode: guarded
  working_dir: .
  heartbeat_interval_sec: 5
  stall_timeout_sec: 120
  kill_on_stall: true
  signal_sources:
    - type: stdout
    - type: log_file_mtime
      paths:
        - .trainee/logs/**/*.log
        - .trainee/runs/**/*.log
  log_paths:
    - .trainee/logs/**/*.log
    - .trainee/runs/**/*.log
  wandb_enabled: false
  tuning_prompt: "Change only one high-impact parameter per round unless the evidence is strong."
```

### Launch environments

`launch.environment` controls how `launch.command` is wrapped:

| Environment | Rendered command prefix |
| --- | --- |
| `system` | `python train.py` |
| `uv` | `uv run python train.py` |
| `venv` | `<project>/.venv/bin/python train.py` when command starts with `python` or `python3` |
| `conda` | `conda run -n <env_name> python train.py` |

Trainee appends arguments in this order:

1. `launch.baseline_config` as `--config <path>`, if set.
2. `launch.args`.
3. `data` entries that have a `flag`.
4. `run.fixed_args`.
5. Agent-controlled `tuning.params`.

Only `tuning.params` may be changed by the agent. `run.fixed_args` stay constant across every round.

For commands that cannot be expressed structurally, use `advanced.shell_command`. Include `{extra_args}` where the generated tunable parameters should be inserted.

Available command-template variables:

- `{project_root}`
- `{working_dir}`
- `{trainee_dir}`
- `{extra_args}`
- `{session_id}`
- `{round_index}`
- `{session_dir}`
- `{round_dir}`
- `{config_path}`

The same round workspace is also exposed through environment variables:

- `TRAINEE_SESSION_ID`
- `TRAINEE_ROUND_INDEX`
- `TRAINEE_SESSION_DIR`
- `TRAINEE_ROUND_DIR`
- `TRAINEE_CONFIG_PATH`

This is useful when a wrapper script needs to write a generated per-round config file.

## Metrics and stopping behavior

Trainee always tries to parse built-in `loss` and `total_loss` values from captured output and configured log paths.

Custom metrics can use:

- `stdout_regex`
- `log_regex`
- `log_file_regex`
- `jsonl`
- `wandb_summary`

Regex metrics should expose the value either as the first capture group or as a named `(?P<value>...)` group.

If `metrics.specs` is empty, at least `loss` or `total_loss` must be found. If required custom metrics are missing, the round is marked `completed_without_metrics` and the session fails.

The first completed round is treated as the baseline. Later rounds are compared against both the baseline and best-so-far round.

If no `tuning.params` are configured, Trainee runs the latest result collection and then stops because there is nothing safe for the agent to change.

## Runtime outputs

By default, project runtime data is stored under the training project’s `.trainee/` directory:

- `.trainee/runtime.sqlite3` — local run database.
- `.trainee/artifacts/session-XXXX/round-XXXX.log` — captured stdout/stderr.
- `.trainee/artifacts/session-XXXX/report.md` — Markdown session report.
- `.trainee/artifacts/session-XXXX/result_ledger.csv` — human-readable experiment ledger.
- `.trainee/artifacts/session-XXXX/result_ledger.jsonl` — machine-readable experiment ledger.
- `.trainee/artifacts/session-XXXX/research_state.json` — baseline, best-so-far, recent rounds, tried changes, and rejected changes.
- `.trainee/runs/session-XXXX/round-XXXX/` — per-round workspace.

Set `TRAINEE_DATA_DIR` to store runtime data elsewhere.

## Guarded vs unsafe execution

The default security mode is `guarded`.

In guarded mode:

- The host filesystem is mounted read-only.
- The training project is read-only.
- The project’s `.trainee/` directory is writable.
- `/tmp` is an isolated tmpfs.
- `HOME`, `XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME`, `MPLCONFIGDIR`, and `WANDB_DIR` are redirected into `.trainee/`.
- Log, signal, and metric file paths must stay inside `.trainee/`.

Use this mode when the training command can write logs/checkpoints/cache under `.trainee/`.

Use unsafe mode when a project is not yet adapted for guarded execution:

```bash
trainee run --unsafe
```

Unsafe mode still redirects common cache/home variables into `.trainee/`, but it does not use `bubblewrap` isolation.

## LLM provider configuration

Trainee supports `none`, `moonshot`, `openai`, and `anthropic`.

For real tuning decisions, configure a provider explicitly:

```bash
export TRAINEE_LLM_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
```

Moonshot:

```bash
export TRAINEE_LLM_PROVIDER=moonshot
export MOONSHOT_API_KEY=...
export MOONSHOT_MODEL=kimi-k2.6
```

Anthropic:

```bash
export TRAINEE_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-3-5-haiku-latest
```

Provider settings can also be edited in the Web UI. They are saved to:

```text
~/.trainee/config.json
```

The global decision system prompt is stored in the same file and can be edited from the Web UI or `/api/runtime/system-prompt`.

If no provider is configured, Trainee uses a limited heuristic fallback. That is useful for smoke tests, but not a replacement for an actual research decision model.

## Prompt and project guidance

The decision prompt includes:

- The global system prompt.
- `.trainee/context.md`.
- `context.md`, if present at the project root.
- `constraints.md`, if present at the project root.
- `metrics.prompt` from `project.yaml`.
- `advanced.tuning_prompt` from `project.yaml`.
- Baseline, best-so-far, recent rounds, tried changes, and rejected changes.

Use `constraints.md` for stable project rules such as “do not change batch size”, “keep evaluation split fixed”, or “only optimize validation MPJPE”.

Use `advanced.tuning_prompt` for run-specific tuning strategy.

## CLI reference

```bash
trainee version
trainee init [project_root] [--baseline-config PATH] [--force]
trainee doctor [project_root]
trainee run [project_root] [--dry-run] [--guarded | --unsafe]
trainee webui [--host HOST] [--port PORT] [--reload] [--no-open]
trainee serve [--host HOST] [--port PORT] [--reload]
trainee tools [--base-url URL] [--name TOOL_NAME]
trainee call TOOL_NAME --input JSON_OR_@FILE_OR_-
trainee report SESSION_ID [--output report.md]
```

Notes:

- Running `trainee` with no subcommand starts the local service, equivalent to `trainee serve`.
- `trainee launch` is kept as a compatibility alias for `trainee init`.
- `trainee run --dry-run` runs preflight checks and prints the baseline command without creating a runtime database.
- `trainee doctor` fails before a session starts if data paths, environment, launcher, sandbox paths, or config validation are not ready.

## Web UI

Start the UI:

```bash
trainee webui
```

The UI can:

- Register or edit a training project.
- Save the same `.trainee/project.yaml` used by the CLI.
- Start, stop, and inspect the loop.
- Preview the next decision prompt.
- Edit provider settings and the global system prompt.
- Save and apply prompt presets.
- Inspect run logs, decisions, agent traces, W&B links, reports, and ledgers.
- Test the configured LLM provider from `/llm-test`.

To start the service without opening a browser:

```bash
trainee serve
```

or:

```bash
trainee webui --no-open
```

## Tool API

Start the local service:

```bash
trainee serve
```

Print OpenAI-style function schemas:

```bash
trainee tools
trainee tools --name loop_start
```

Call a tool:

```bash
trainee call loop_get
trainee call runs_get --input '{"run_id": 1}'
trainee call project_get
```

Common tools:

- `project_register`
- `project_get`
- `project_update_context`
- `runtime_provider_get`
- `runtime_provider_update`
- `runtime_debug_get`
- `runtime_debug_update`
- `runtime_system_prompt_get`
- `runtime_system_prompt_update`
- `prompt_preview`
- `prompt_presets_list`
- `prompt_presets_save`
- `loop_start`
- `loop_get`
- `loop_stop`
- `runs_list`
- `runs_get`
- `session_report`

`project_register` accepts the same structure as `.trainee/project.yaml` plus a top-level `project_root`. It writes the normalized config back to that project.

## Formal-use checklist

Before using Trainee on a real training project:

1. Commit or otherwise save the training project state.
2. Run `trainee init` and review `.trainee/project.yaml`.
3. Confirm `data` paths exist and stay inside the project root.
4. Confirm `launch.command`, `launch.baseline_config`, `launch.args`, and `run.fixed_args` reproduce the intended baseline command.
5. Confirm only safe parameters are listed in `tuning.params`.
6. Confirm metrics can be parsed from stdout, `.trainee/` logs, JSONL, or W&B summary.
7. For guarded mode, make the training job write logs, checkpoints, W&B files, and caches under `.trainee/`.
8. Run `trainee doctor`.
9. Run `trainee run --dry-run` and inspect the printed command.
10. Configure an LLM provider if you expect non-trivial tuning decisions.
11. Start with a small `max_rounds` and short timeout before increasing the budget.

## Development

Run tests:

```bash
uv run pytest
```

The package entrypoint is:

```text
trainee = trainee.cli:main
```
