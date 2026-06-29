# Trainee 中文说明

[English README](README.md)

<img width="256" height="256" alt="Trainee logo" src="https://github.com/user-attachments/assets/51a0b054-5081-4862-8d38-d3601fd3d699" />

Trainee 是一个用于自动化外部模型训练循环的保守型 agent runtime。

它不包含你的训练代码。Trainee 的职责是连接到已有训练项目，执行项目自己的训练命令，观察训练进度，解析训练指标，再调用已配置的 LLM 生成下一轮参数，直到循环停止。

典型流程：

```text
读取项目上下文 -> 运行训练 -> 监控 activity -> 解析指标
-> 对比 baseline / best-so-far -> 决定下一组参数 -> 继续或停止
```

## 功能概览

- 通过 `.trainee/project.yaml` 初始化和管理训练项目。
- 自动发现可能的入口脚本、数据目录、配置文件、环境类型和固定训练参数。
- 支持 `system`、`uv`、`.venv`、`conda` 等运行环境。
- 支持基于 `bubblewrap` 的 guarded 执行模式：项目只读，仅 `.trainee/` 可写。
- 从 stdout、stderr、日志文件修改时间、heartbeat JSON 等来源被动监控训练活动。
- 从 stdout 正则、日志正则、JSONL、W&B summary 等来源提取指标。
- 记录 baseline、best-so-far、假设、变更摘要和已拒绝变更，避免重复无效实验。
- 导出 Markdown、CSV、JSONL、JSON 格式的 session 报告和实验 ledger。
- 提供本地 Web UI 和 HTTP Tool API，用于项目配置、循环控制、prompt 预览、运行检查、provider 设置和报告查看。

## 环境要求

- Python 3.9 或更新版本。
- 推荐在 Linux 上使用 guarded 模式。
- guarded 模式需要安装 `bubblewrap` / `bwrap`。
- 推荐使用 `uv` 安装，但 Trainee 本身也是普通 Python package。
- 需要一个可以从终端运行的外部训练项目。

如果系统没有 `bwrap`，或者训练任务必须写入 `.trainee/` 之外的目录，需要显式使用 `--unsafe`。

## 安装

在 Trainee 仓库中执行：

```bash
uv tool install --editable . --force
uv tool update-shell
```

必要时重启 shell，然后验证：

```bash
trainee version
trainee --help
```

本地开发时也可以不全局安装：

```bash
uv run trainee --help
```

## 快速开始

进入你希望 Trainee 控制的训练项目：

```bash
cd /path/to/training-project
trainee init
```

如果已有 baseline 配置文件：

```bash
trainee init --baseline-config configs/base.yaml
```

然后编辑生成的配置：

```bash
$EDITOR .trainee/project.yaml
$EDITOR .trainee/tuning.yaml
```

运行前先检查项目和最终 baseline 命令：

```bash
trainee doctor
trainee run --dry-run
```

启动训练循环：

```bash
trainee run
```

启动本地 Web UI：

```bash
trainee webui
```

如果浏览器没有自动打开，访问：

```text
http://127.0.0.1:8000
```

## 项目初始化

`trainee init` 会创建：

- `.trainee/project.yaml`：主项目运行配置。
- `.trainee/tuning.yaml`：允许 Trainee 调整的参数白名单。
- `.trainee/context.md`：Trainee 生成的项目上下文理解。
- `.trainee/README.md`：检测到的入口、配置、目录等候选项说明。
- `.trainee/logs/`、`.trainee/runs/`、`.trainee/artifacts/`：运行时输出目录。

初始化默认不破坏已有文件：

```bash
trainee init
trainee init --baseline-config configs/base.yaml
trainee prepare
trainee init --force
```

`--baseline-config` 必须指向项目内已存在的文件。Trainee 会把它记录为 `launch.baseline_config`，每轮运行时复制到 `.trainee/runs/session-XXXX/round-XXXX/config.yaml`，再通过 `--config <path>` 传给训练命令。

`trainee init` 不会直接推断所有可调参数或输出路径。需要在 `.trainee/project.yaml` 里填写 `output.config_path`，指向 baseline 配置中控制训练输出目录的字段，例如 `output.root`，然后运行：

```bash
trainee prepare
```

`prepare` 会读取项目上下文和 baseline 配置，校验字段，并在 `.trainee/tuning.yaml` 为空时生成保守的候选可调参数。运行训练前必须人工检查生成结果。

## 配置文件

`.trainee/project.yaml` 和 `.trainee/tuning.yaml` 是 CLI、Web UI 和 Tool API 的共同配置来源。运行时 Trainee 会加载两个文件，并编译成一个不可变的 session 配置。

示例 `.trainee/project.yaml`：

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

output:
  config_path: output.root

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

`run.timeout_minutes` 是唯一会硬终止训练进程的运行中条件。Activity monitor 只更新 UI 状态，不会 kill 进程，也不会把 round 标记为失败。

示例 `.trainee/tuning.yaml`：

```yaml
version: 1

params:
  - name: lr
    flag: --lr
    type: float
    default: 0.001
    min_value: 0.00001
    max_value: 0.01
  - name: theta_weight
    config_path: fit.term_weights.theta
    type: float
    min_value: 1.0
    max_value: 15.0
```

只有 `.trainee/tuning.yaml` 中列出的参数允许被 agent 修改。`launch.args` 和 `run.fixed_args` 在每轮中保持不变，并会从 tunable discovery 中排除。

对于写在 baseline 配置文件里的参数，使用 `params[].config_path`。Trainee 会在每轮生成的 config 中写入新值，不会把它追加成 CLI 参数。

## 启动环境

`launch.environment` 决定如何包装 `launch.command`：

| Environment | 最终命令前缀 |
| --- | --- |
| `system` | `python train.py` |
| `uv` | `uv run python train.py` |
| `venv` | 当命令以 `python` 或 `python3` 开头时，使用 `<project>/.venv/bin/python train.py` |
| `conda` | `conda run -n <env_name> python train.py` |

Trainee 会按以下顺序追加参数：

1. 如果设置了 `launch.baseline_config`，先生成每轮配置文件，并作为 `--config <path>` 传入。
2. `launch.args`。
3. 带有 `flag` 的 `data` 配置。
4. `run.fixed_args`。
5. `.trainee/tuning.yaml` 中由 agent 控制的 CLI 参数。

如果命令无法用结构化配置表达，可以使用 `advanced.shell_command`。在命令中包含 `{extra_args}`，表示 Trainee 生成的可调参数插入位置。

常用模板变量：

- `{project_root}`
- `{working_dir}`
- `{trainee_dir}`
- `{extra_args}`
- `{session_id}`
- `{round_index}`
- `{session_dir}`
- `{round_dir}`
- `{config_path}`

同一轮 workspace 也会暴露为环境变量：

- `TRAINEE_SESSION_ID`
- `TRAINEE_ROUND_INDEX`
- `TRAINEE_SESSION_DIR`
- `TRAINEE_ROUND_DIR`
- `TRAINEE_CONFIG_PATH`

## 指标与停止行为

Trainee 默认会尝试从捕获的输出和已配置日志中解析内置的 `loss` 和 `total_loss`。

自定义指标支持：

- `stdout_regex`
- `log_regex`
- `log_file_regex`
- `jsonl`
- `wandb_summary`

正则指标需要把数值放在第一个捕获组，或命名为 `(?P<value>...)`。

如果 `metrics.specs` 为空，Trainee 至少需要找到 `loss` 或 `total_loss`。如果必需的自定义指标缺失，本轮会被标记为 `completed_without_metrics`，session 会失败。

第一个完成的 round 会被视为 baseline。之后的 round 会同时和 baseline、best-so-far 比较。

如果 `.trainee/tuning.yaml` 没有任何参数，Trainee 会收集最新结果后停止，因为没有安全可改的参数。

## 运行输出

默认情况下，运行数据存放在训练项目的 `.trainee/` 目录：

- `.trainee/project.yaml`、`.trainee/tuning.yaml`、`.trainee/context.md`：项目级 Trainee 文件。
- `.trainee/runtime.sqlite3`：本地运行数据库。
- `.trainee/artifacts/session-XXXX/round-XXXX.log`：捕获的 stdout/stderr。
- `.trainee/artifacts/session-XXXX/report.md`：Markdown session 报告。
- `.trainee/artifacts/session-XXXX/result_ledger.csv`：便于人工阅读的实验 ledger。
- `.trainee/artifacts/session-XXXX/result_ledger.jsonl`：便于机器读取的实验 ledger。
- `.trainee/artifacts/session-XXXX/research_state.json`：baseline、best-so-far、recent rounds、tried changes、rejected changes。
- `.trainee/runs/session-XXXX/round-XXXX/`：每轮 workspace。

如果直接运行 `trainee`、`trainee serve` 或 `trainee webui` 且没有绑定训练项目，服务运行数据会存放在 `~/.trainee/runtime/`。

可以通过 `TRAINEE_DATA_DIR` 把运行数据存到其他位置。它只改变 runtime 数据库和 artifact 的位置，不改变项目配置文件的位置；项目配置仍保存在项目内 `.trainee/`。

## Guarded 与 Unsafe 模式

默认安全模式是 `guarded`。

在 guarded 模式下：

- host 文件系统只读挂载。
- 训练项目只读。
- 只有项目内 `.trainee/` 可写。
- `/tmp` 是隔离的 tmpfs。
- `HOME`、`XDG_CACHE_HOME`、`HF_HOME`、`TORCH_HOME`、`MPLCONFIGDIR`、`WANDB_DIR` 会被重定向到 `.trainee/`。
- 日志、信号和指标文件路径必须位于 `.trainee/` 内。

当训练命令可以把日志、checkpoint、cache 写到 `.trainee/` 下时，优先使用 guarded 模式。

如果项目还没有适配 guarded 执行，可以显式使用 unsafe 模式：

```bash
trainee run --unsafe
```

unsafe 模式仍会把常见 cache/home 变量重定向到 `.trainee/`，但不会使用 `bubblewrap` 隔离。

## LLM Provider 配置

Trainee 支持 `none`、`moonshot`、`openai` 和 `anthropic`。

真实调参决策需要显式配置 provider。

OpenAI：

```bash
export TRAINEE_LLM_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
```

Moonshot：

```bash
export TRAINEE_LLM_PROVIDER=moonshot
export MOONSHOT_API_KEY=...
export MOONSHOT_MODEL=kimi-k2.6
```

Anthropic：

```bash
export TRAINEE_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-3-5-haiku-latest
```

Provider 设置也可以在 Web UI 中编辑，并保存到：

```text
~/.trainee/config.json
```

全局 decision system prompt 也保存在同一文件中，可以通过 Web UI 或 `/api/runtime/system-prompt` 编辑。

`init`、`prepare`、`doctor` 和 `run` 会通过 live API request 检查 provider 可用性。离线配置时可以对 `init`、`prepare`、`doctor` 使用 `--skip-provider-test`，但 `run` 总是需要可用 provider。

## Prompt 与项目约束

决策 prompt 会包含：

- 全局 system prompt。
- `.trainee/context.md`。
- 项目根目录下的 `context.md`，如果存在。
- 项目根目录下的 `constraints.md`，如果存在。
- `project.yaml` 中的 `metrics.prompt`。
- `project.yaml` 中的 `advanced.tuning_prompt`。
- baseline、best-so-far、recent rounds、tried changes、rejected changes。

建议把稳定项目规则写入 `constraints.md`，例如“不要修改 batch size”、“保持验证集固定”或“只优化 validation MPJPE”。

`advanced.tuning_prompt` 更适合写本次运行特定的调参策略。

## CLI 参考

```bash
trainee version
trainee init [project_root] [--baseline-config PATH] [--force] [--skip-provider-test]
trainee prepare [project_root] [--replace] [--skip-provider-test]
trainee tunables discover [project_root] [--apply] [--replace] [--limit N]
trainee doctor [project_root] [--skip-provider-test]
trainee run [project_root] [--dry-run] [--guarded | --unsafe]
trainee webui [project_root] [--host HOST] [--port PORT] [--reload] [--no-open]
trainee serve [project_root] [--host HOST] [--port PORT] [--reload]
trainee tools [--base-url URL] [--name TOOL_NAME]
trainee call TOOL_NAME --input JSON_OR_@FILE_OR_-
trainee report SESSION_ID [--output report.md]
```

说明：

- 直接运行 `trainee` 且不带子命令时，会启动本地服务，等价于 `trainee serve`。
- `trainee run --dry-run` 会执行 preflight 检查并打印 baseline 命令，不创建运行数据库。
- `trainee doctor` 会在 session 启动前检查数据路径、运行环境、launcher、sandbox 路径、provider live test 和配置有效性。

## Web UI

启动 UI：

```bash
trainee webui
```

绑定到某个训练项目：

```bash
trainee webui /path/to/training-project
```

Web UI 可以：

- 注册或编辑训练项目。
- 保存 CLI 共用的 `.trainee/project.yaml` 和 `.trainee/tuning.yaml`。
- 启动、停止和检查训练循环。
- 预览下一次 decision prompt。
- 编辑 provider 设置和全局 system prompt。
- 保存和应用 prompt presets。
- 查看运行日志、决策、agent trace、W&B 链接、报告和 ledger。
- 在 `/llm-test` 或 provider settings 面板测试 LLM provider。

只启动服务且不打开浏览器：

```bash
trainee serve /path/to/training-project
```

或：

```bash
trainee webui /path/to/training-project --no-open
```

## Tool API

启动本地服务：

```bash
trainee serve
```

打印 OpenAI-style function schemas：

```bash
trainee tools
trainee tools --name loop_start
```

调用工具：

```bash
trainee call loop_get
trainee call runs_get --input '{"run_id": 1}'
trainee call project_get
```

常用 tools：

- `project_register`
- `project_get`
- `project_update_context`
- `runtime_provider_get`
- `runtime_provider_update`
- `runtime_provider_test`
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

`project_register` 接收项目字段、`tuning` 和顶层 `project_root`，并把规范化后的 `.trainee/project.yaml` 与 `.trainee/tuning.yaml` 写回项目。

## 正式使用前检查清单

1. 提交或保存训练项目当前状态。
2. 运行 `trainee init`，检查 `.trainee/project.yaml` 和 `.trainee/tuning.yaml`。
3. 确认 `data` 路径存在，并位于项目根目录内。
4. 确认 `launch.command`、`launch.baseline_config`、`launch.args` 和 `run.fixed_args` 能复现预期 baseline 命令。
5. 运行 `trainee prepare`，检查生成的输出配置和可调参数。
6. 确认 `.trainee/tuning.yaml` 中只包含安全可调参数。
7. 确认指标可以从 stdout、`.trainee/` 日志、JSONL 或 W&B summary 中解析。
8. 如果使用 guarded 模式，确保训练任务把日志、checkpoint、W&B 文件和 cache 写到 `.trainee/` 下。
9. 运行 `trainee doctor`。
10. 运行 `trainee run --dry-run`，检查打印出的命令。
11. 配置并 live-test LLM provider。
12. 先用较小的 `max_rounds` 和较短 timeout 试跑，再增加预算。

## 开发

运行测试：

```bash
uv run pytest
```

package entrypoint：

```text
trainee = trainee.cli:main
```
