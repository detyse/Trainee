# Trainee

保守的自动训练 agent

Trainee 是一个面向训练自动化的 agent runtime。它不承载训练代码本身，而是接入外部训练项目与外部训练环境，围绕“读上下文 -> 发起训练 -> 监控 heartbeat -> 解析结果 -> 决定下一轮参数 -> 自动继续或停止”这条闭环，提供一个最小可用的 runtime 和控制台。

## 能力范围

- 注册一个本机外部训练项目，并用 `.trainee/project.yaml` 保存数据、环境、固定运行预算、可调参数和指标规则。
- 自动扫描外部项目，生成 `ProjectContext`，把“项目是什么、数据在哪、训练入口在哪、参数怎么调、结果怎么看”固化进上下文。
- 使用显式 launcher 执行训练，例如 `conda run ... python train.py ...` 或 `/path/to/train.sh ...`。
- 默认用 bubblewrap 进行 guarded run：宿主文件系统只读，只有项目内 `.trainee/` 可写。
- 通过本地子进程输出、日志文件更新时间等 signal source 做 heartbeat 检测；超过阈值则标记 stalled。
- 通过 stdout regex、显式日志文件 regex、JSONL 等 metric source 抽取 `loss/total_loss` 和自定义指标。
- 通过轻量控制台查看项目信息、loop 状态、轮次历史、命令、参数、日志路径、W&B 链接和 agent 决策。

## 快速开始

1. agent 安装：

```bash
cd /path/to/trainee
uv tool install -e .
```

2. 在训练项目中初始化：

```bash
cd /path/to/training-project
trainee init
```

3. 编辑 `.trainee/project.yaml`，然后检查最终 baseline 命令：

```bash
trainee doctor
trainee run --dry-run
```

4. 开始运行：

```bash
trainee run
```

如需网页配置和运行监控，执行 `trainee webui` 并打开 `http://127.0.0.1:8000`。

## 全局 CLI 安装

如果希望在任意目录直接使用 `trainee`，推荐用 `uv tool install` 做用户级全局安装：

```bash
cd /path/to/Trainee
uv tool install --editable . --force
uv tool update-shell
```

重启 shell 后验证：

```bash
trainee --help
trainee version
trainee webui --host 127.0.0.1 --port 8000
```

`trainee version` 会输出当前版本和该版本的最后更新日期。

`trainee webui` 会启动本地服务并打开浏览器页面；只想启动服务、不打开浏览器时使用 `trainee serve`，或运行 `trainee webui --no-open`。

`--editable` 会让全局命令直接指向当前源码目录。后续修改或拉取代码后，通常不需要重新安装，`trainee` 会直接使用新代码。

如果 shell 还找不到 `trainee`，临时加入 PATH：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

也可以直接使用完整路径：

```bash
$HOME/.local/bin/trainee --help
```

Provider 配置默认写入 `$HOME/.trainee/config.json`。项目运行数据写入项目内 `.trainee/runtime.sqlite3` 和 `.trainee/artifacts/`；如果希望运行数据放到别的位置，可以设置 `TRAINEE_DATA_DIR`。

## 项目初始化

在目标训练项目目录执行：

```bash
cd ~/project/toymodel
trainee init
```

`init` 是项目初始化动作，不启动 Web UI。它会生成一份可编辑的项目草稿，而不是直接开始训练。它会像 agent 一样在终端输出本次读了哪些文件、写了哪些文件，例如：

```text
Trainee init
- Project: /home/user/project/toymodel
- Status: initialized new project files

Files
- Read: README.md
- Read: train.py
- Wrote: .trainee/project.yaml
- Wrote: .trainee/context.md
- Wrote: .trainee/program.md
- Wrote: .trainee/README.md
- Config: .trainee/project.yaml

Discovery
- Environment: conda (trainer)
- Entrypoints: train.py
- Data candidates: data
- Config candidates: configs/base.yaml
- Training limit candidates: --max-iter=1000

Effective configuration
- Environment: conda (trainer)
- Working directory: /home/user/project/toymodel
- Security: guarded
- Budget: max_rounds=3, timeout=60 minutes
- Data inputs: data
- Launch arguments: --config=configs/base.yaml
- Fixed arguments: --max-iter=1000
- Tunable parameters: none
- Metrics: none (built-in loss/total_loss parsing only)
- Runtime: kill_on_stall=true, wandb=disabled
- Heartbeat: every 5s, stall after 120s; sources=stdout; log_file_mtime(.trainee/logs/**/*.log, .trainee/runs/**/*.log)
- Log paths: .trainee/logs/**/*.log, .trainee/runs/**/*.log
- Launcher: conda run -n trainer python train.py --config configs/base.yaml --max-iter 1000 {extra_args}

Next
- Review: .trainee/project.yaml, .trainee/context.md, and .trainee/program.md
- Validate: trainee doctor or trainee run --dry-run
- Next: edit .trainee/project.yaml, run `trainee doctor`, then run `trainee run`
```

项目内生成的 `.trainee/` 用来保存这个项目的初始化草稿和上下文说明：

- `.trainee/project.yaml`: 唯一面向用户的运行配置。
- `.trainee/context.md`: Trainee 读取项目代码后生成的项目理解。
- `.trainee/program.md`: 每轮 decision 都会使用的固定 agent rules，可直接编辑，也可在 Web UI 的 Prompt 页修改。
- `.trainee/README.md`: 本目录文件说明。

推荐工作流：

1. 编辑 `.trainee/project.yaml`，确认数据、环境、训练命令、固定限制、可调参数和指标。
2. 检查 `.trainee/program.md` 与 `.trainee/context.md`，补充 agent 规则和项目目标。
3. 运行 `trainee doctor` 或 `trainee run --dry-run`，检查完整 baseline 命令。
4. 确认无误后运行 `trainee run`。

如果文件已存在，`trainee init` 默认保留旧文件；需要重写时使用：

```bash
trainee init --force
```

`init` 会分别输出自动探测到的候选信息和最终生效配置，包括环境、训练入口、数据目录、配置文件、固定预算参数、可调参数、指标、heartbeat 与完整 launcher，并把候选项写入 `.trainee/README.md`。默认使用 guarded 模式、stdout heartbeat 和 3 个 round。

## 项目运行

在目标训练项目目录执行：

```bash
trainee run
```

`trainee run` 会先执行与 `trainee doctor` 相同的 preflight。数据路径、训练入口、环境或 sandbox 检查失败时不会创建 session 或启动训练。检查通过后，它读取 `.trainee/project.yaml`，使用项目本地 `.trainee/runtime.sqlite3` 和 `.trainee/artifacts/` 启动 loop。

只检查配置和最终命令：

```bash
trainee run --dry-run
```

训练子进程在 guarded 模式下只能写项目 `.trainee/`，并会把 `HOME`、`XDG_CACHE_HOME`、`WANDB_DIR`、`HF_HOME`、`TORCH_HOME`、`MPLCONFIGDIR` 重定向到 `.trainee/` 下。需要临时绕过 sandbox 时显式使用：

```bash
trainee run --unsafe
```

普通 `trainee serve` 不绑定当前目录，也不会自动初始化当前目录。项目初始化只通过 `trainee init` 显式发生。`trainee launch` 暂时保留为兼容 alias。

## 后续更新

更新代码：

```bash
cd /path/to/Trainee
git status
git pull --ff-only origin main
```

如果只改了 Python 源码，editable 全局 CLI 会自动使用新代码。

如果更新涉及依赖、`pyproject.toml`、命令入口或静态资源打包配置，重新安装一次全局 CLI：

```bash
cd /path/to/Trainee
uv tool install --editable . --force
```

然后验证：

```bash
trainee --help
```

更新前如果有本地修改，先提交或暂存，避免 `git pull` 冲突：

```bash
git add .
git commit -m "local changes"
```

或：

```bash
git stash push -u
```

## 工具式调用

除了网页控制台，也可以把 Trainee 当作本地工具服务调用。先启动服务：

```bash
trainee serve
```

查看可用工具和 JSON Schema：

```bash
trainee tools
```

工具清单使用 OpenAI-style function schema 形状，工具名只包含字母、数字和下划线，方便直接交给 agent 编排。常用工具：

- `project_register`
- `project_get`
- `project_get_program`
- `project_update_program`
- `loop_start`
- `loop_get`
- `loop_stop`
- `runs_list`
- `runs_get`
- `prompt_preview`

`project_register` 接收与 `project.yaml` 相同的结构，并额外要求顶层 `project_root`。注册后会写回该项目的 `.trainee/project.yaml`，因此 CLI、Web UI 和工具调用共享同一个配置来源。

```bash
trainee call project_register --input @registration.json
trainee call loop_start
trainee call loop_get
trainee call runs_list
trainee call runs_get --input '{"run_id": 1}'
```

如果没有安装全局 CLI，把以上命令里的 `trainee` 替换为 `uv run trainee`。

## project.yaml

```yaml
version: 1

data:
  - path: data/train
    flag: --data-root

launch:
  environment: conda       # system | uv | venv | conda
  env_name: trainer
  command: [python, train.py]
  args:
    - flag: --config
      value: configs/base.yaml

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
      max_value: 0.1

metrics:
  specs:
    - name: total_loss
      source: stdout_regex
      key_or_pattern: 'total_loss=(?P<value>-?\d+(?:\.\d+)?)'
      goal: min
      required: true

advanced: {}
```

- `data`: 数据路径；相对路径以项目根目录解析。可选 `flag` 会把路径作为固定训练参数传入。
- `launch`: 环境和训练命令。`uv` 自动添加 `uv run`，`venv` 使用 `.venv/bin/python`，`conda` 使用 `conda run -n ENV`。
- `run.fixed_args`: 每轮固定不变，例如 `--max-iter`、`--limit` 或固定 config。
- `tuning.params`: agent 唯一可以修改的参数白名单。
- `metrics.specs`: 指标抽取规则。
- `advanced`: 可覆盖 `security_mode`、`working_dir`、heartbeat、signal sources、W&B 和 `shell_command`。

`metrics.specs.source` 常用值：

- `stdout_regex`: 只从 Trainee 捕获的子进程 stdout/stderr 内部日志中按正则抽取。
- `log_file_regex`: 从 `path` 或 `paths` 指定的日志文件/glob 中按正则抽取；glob 会在评估时重新展开。
- `jsonl`: 从 `path` 或 `paths` 指定的 JSONL 文件中读取最后一个可解析数值，`key_or_pattern` 为字段名或点分路径。
- `log_regex`: 旧版兼容来源，会读取内部日志和 `log_paths`。

## 全局配置与 LLM Provider

Trainee 的项目初始化目录和全局配置目录是分开的：

- 在训练项目目录执行 `trainee init`，例如 `cd ~/project/toymodel && trainee init`，Trainee 会读取当前目录并生成项目内 `.trainee/` 文件。
- 普通 `trainee serve` 不读取当前目录，也不会自动初始化当前目录。
- Provider 全局配置默认放在 `~/.trainee/config.json`。
- 项目运行数据默认放在项目内 `.trainee/runtime.sqlite3` 和 `.trainee/artifacts/`；`TRAINEE_DATA_DIR` 只覆盖运行数据目录，不影响 Provider 配置位置。
- Provider 设置优先级为：进程环境变量最高，`~/.trainee/config.json` 其次。
- 项目不读取 `.env` 文件；Web UI 保存 provider 设置时只写 `~/.trainee/config.json`。
- Agent Debug 默认关闭；可在 Runtime 页面开启，开启后后续 round 会保存 provider 原始响应、解析/校验错误和 heuristic fallback 原因。运行中的 loop 不允许切换该开关。

未配置 LLM key 时，runtime 会自动回退到启发式调参。

`~/.trainee/config.json` Moonshot / Kimi 示例：

```json
{
  "agent_debug_enabled": false,
  "llm_provider": "moonshot",
  "llm_timeout_sec": 30,
  "moonshot": {
    "api_key": "sk-...",
    "base_url": "https://api.moonshot.cn/v1",
    "model": "kimi-k2.6"
  }
}
```

`~/.trainee/config.json` OpenAI-compatible 示例：

```json
{
  "llm_provider": "openai",
  "llm_timeout_sec": 30,
  "openai": {
    "api_key": "sk-...",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini"
  }
}
```

`~/.trainee/config.json` Claude / Anthropic 示例：

```json
{
  "llm_provider": "anthropic",
  "llm_timeout_sec": 30,
  "anthropic": {
    "api_key": "sk-ant-...",
    "base_url": "https://api.anthropic.com",
    "model": "claude-3-5-sonnet-latest",
    "version": "2023-06-01",
    "max_tokens": 1024
  }
}
```

如果不显式设置 `TRAINEE_LLM_PROVIDER`，runtime 会自动优先使用已存在的 key：

- 有 `MOONSHOT_API_KEY` 时走 `moonshot`
- 有 `OPENAI_API_KEY` 时走 `openai`
- 只有 `ANTHROPIC_API_KEY` 时走 `anthropic`
- 这些 key 都没有时禁用 LLM，回退到启发式策略

## LLM 决策

- `moonshot` provider 调用 `MOONSHOT_BASE_URL/chat/completions`，默认使用 `kimi-k2.6` 生成结构化 `AgentDecision`。
- `openai` provider 调用 `OPENAI_BASE_URL/chat/completions`，使用 `OPENAI_MODEL` 生成结构化 `AgentDecision`。
- `anthropic` provider 调用 `ANTHROPIC_BASE_URL/v1/messages`，使用 Claude Messages API 生成结构化 `AgentDecision`。
- 如果没有配置 LLM，runtime 会回退到一个保守的启发式策略，优先调整 `lr` / `learning_rate` 一类参数。


## 测试

```bash
pytest
```
