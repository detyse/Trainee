# Trainee

Trainee 是一个面向训练自动化的 agent runtime。它不承载训练代码本身，而是接入外部训练项目与外部训练环境，围绕“读上下文 -> 发起训练 -> 监控 heartbeat -> 解析结果 -> 决定下一轮参数 -> 自动继续或停止”这条闭环，提供一个最小可用的 runtime 和控制台。

## 能力范围

- 注册一个本机外部训练项目，并保存项目根目录、工作目录、启动命令模板、数据路径、日志路径、可调参数和指标规则。
- 自动扫描外部项目，生成 `ProjectContext`，把“项目是什么、数据在哪、训练入口在哪、参数怎么调、结果怎么看”固化进上下文。
- 使用显式 launcher 执行训练，例如 `conda run ... python train.py ...` 或 `/path/to/train.sh ...`。
- 默认用 bubblewrap 进行 guarded run：宿主文件系统只读，只有项目内 `.trainee/` 可写。
- 通过本地子进程输出和日志更新时间做 heartbeat 检测；超过阈值则标记 stalled。
- 解析日志里的 `loss/total_loss`，并支持按配置从正则或 W&B summary 中抽取更多指标。
- 通过轻量控制台查看项目信息、loop 状态、轮次历史、命令、参数、日志路径、W&B 链接和 agent 决策。

## 快速开始

1. 安装依赖：

```bash
python3 -m pip install -e ".[dev]"
```

2. 启动服务：

```bash
python3 main.py
```

3. 打开 `http://127.0.0.1:8000`

4. 填写外部训练项目配置。`launcher_template` 可以直接写完整命令，例如：

```bash
conda run -n trainer python {project_root}/train.py --config {project_root}/configs/base.yaml {extra_args}
```

支持的模板变量：

- `{project_root}`
- `{working_dir}`
- `{extra_args}`

`{extra_args}` 由 runtime 根据 `tunable_params` 自动拼成 CLI 参数。agent 不会自由生成新的 shell 片段，只能覆盖白名单参数。

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
trainee webui --host 127.0.0.1 --port 8000
```

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

默认运行数据写入 `$HOME/.trainee`，包括 `runtime.sqlite3`、`artifacts/` 和 `config.json`。如果希望放到别的位置，可以设置 `TRAINEE_DATA_DIR`。

## 项目初始化

在目标训练项目目录执行：

```bash
cd ~/project/toymodel
trainee init
```

`init` 是项目初始化动作，不启动 Web UI。它会像 agent 一样在终端输出本次读了哪些文件、写了哪些文件，例如：

```text
Trainee init
- Project: /home/user/project/toymodel
- Read: README.md
- Read: train.py
- Wrote: .trainee/project.json
- Wrote: .trainee/context.md
- Wrote: .trainee/README.md
- Launcher: python {project_root}/train.py {extra_args}
```

项目内生成的 `.trainee/` 用来保存这个项目的初始化草稿和上下文说明：

- `.trainee/project.json`: 可编辑的项目注册草稿。
- `.trainee/context.md`: Trainee 读取项目代码后生成的项目理解。
- `.trainee/README.md`: 本目录文件说明。

如果文件已存在，`trainee init` 默认保留旧文件；需要重写时使用：

```bash
trainee init --force
```

生成的 `project.json` 默认 `security_mode` 为 `guarded`，默认 `log_paths` 指向 `.trainee/logs/**/*.log` 和 `.trainee/runs/**/*.log`。Launcher 模板可用变量：

- `{project_root}`
- `{working_dir}`
- `{trainee_dir}`
- `{extra_args}`

## 项目运行

在目标训练项目目录执行：

```bash
trainee run
```

`trainee run` 会读取 `.trainee/project.json`，使用项目本地 `.trainee/runtime.sqlite3` 和 `.trainee/artifacts/`，启动 loop 并在当前终端等待完成。默认 guarded 模式要求系统安装 `bubblewrap`/`bwrap`；如果没有 `bwrap`，运行会失败并提示安装。

训练子进程在 guarded 模式下只能写项目 `.trainee/`，并会把 `HOME`、`XDG_CACHE_HOME`、`WANDB_DIR`、`HF_HOME`、`TORCH_HOME`、`MPLCONFIGDIR` 重定向到 `.trainee/` 下。需要临时绕过 sandbox 时显式使用：

```bash
trainee run --unsafe
```

普通 `trainee serve` 不绑定当前目录，只使用全局 runtime/settings。项目初始化只通过 `trainee init` 显式发生。`trainee launch` 暂时保留为兼容 alias。

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
- `loop_start`
- `loop_get`
- `loop_stop`
- `runs_list`
- `runs_get`
- `prompt_preview`

注册项目可以准备一个 `project.json`：

```json
{
  "project_root": "/path/to/external-project",
  "working_dir": "/path/to/external-project",
  "launcher_template": "python {project_root}/train.py {extra_args}",
  "security_mode": "guarded",
  "data_paths": ["/path/to/external-project/data"],
  "log_paths": ["/path/to/external-project/.trainee/logs/*.log"],
  "heartbeat_interval_sec": 5,
  "stall_timeout_sec": 120,
  "max_rounds": 3,
  "tunable_params": [
    {
      "name": "lr",
      "flag": "--lr",
      "type": "float",
      "default": 0.1,
      "min_value": 0.001,
      "max_value": 1.0
    }
  ],
  "metric_specs": [
    {
      "name": "total_loss",
      "source": "log_regex",
      "key_or_pattern": "total_loss=(?P<value>-?\\d+(?:\\.\\d+)?)",
      "goal": "min",
      "required": true
    }
  ]
}
```

然后按工具名调用：

```bash
trainee call project_register --input @project.json
trainee call loop_start
trainee call loop_get
trainee call runs_list
trainee call runs_get --input '{"run_id": 1}'
```

也可以通过 stdin 传入 JSON：

```bash
printf '%s\n' '{"run_id": 1}' | trainee call runs_get --input -
```

如果没有安装全局 CLI，把以上命令里的 `trainee` 替换为 `uv run trainee`。

## 配置字段

- `project_root`: 外部训练项目根目录
- `working_dir`: 训练命令实际执行目录
- `launcher_template`: 完整启动命令模板
- `security_mode`: `guarded` 或 `unsafe`，默认 `guarded`
- `data_paths`: 数据路径列表，JSON 数组
- `log_paths`: 外部日志路径或 glob 列表，JSON 数组
- `tunable_params`: 可调参数白名单，JSON 数组
- `metric_specs`: 训练结果指标规则，JSON 数组
- `metric_prompt`: 给 agent 的指标解读提示
- `tuning_prompt`: 给 agent 的调参提示

`tunable_params` 示例：

```json
[
  {
    "name": "lr",
    "flag": "--lr",
    "type": "float",
    "default": 0.1,
    "min_value": 0.001,
    "max_value": 1.0
  },
  {
    "name": "epochs",
    "flag": "--epochs",
    "type": "int",
    "default": 3,
    "min_value": 1,
    "max_value": 20
  }
]
```

`metric_specs` 示例：

```json
[
  {
    "name": "total_loss",
    "source": "log_regex",
    "key_or_pattern": "total_loss=(?P<value>-?\\d+(?:\\.\\d+)?)",
    "goal": "min",
    "required": true
  }
]
```

## 全局配置与 LLM Provider

Trainee 的项目初始化目录和全局配置目录是分开的：

- 在训练项目目录执行 `trainee init`，例如 `cd ~/project/toymodel && trainee init`，Trainee 会读取当前目录并生成项目内 `.trainee/` 文件。
- 普通 `trainee serve` 不读取当前目录，只打开全局 runtime/settings。
- 全局运行数据默认放在 `~/.trainee`，包括 `runtime.sqlite3`、`artifacts/` 和 `config.json`。
- Provider 设置优先级为：进程环境变量最高，`~/.trainee/config.json` 其次。
- 项目不读取 `.env` 文件；Web UI 保存 provider 设置时只写 `~/.trainee/config.json`。

未配置 LLM key 时，runtime 会自动回退到启发式调参。

`~/.trainee/config.json` Moonshot / Kimi 示例：

```json
{
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

相关环境变量：

- `TRAINEE_DATA_DIR`
- `TRAINEE_LLM_PROVIDER`
- `LLM_PROVIDER`
- `MOONSHOT_API_KEY`
- `MOONSHOT_BASE_URL`
- `MOONSHOT_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_MODEL`
- `ANTHROPIC_VERSION`
- `ANTHROPIC_MAX_TOKENS`
- `TRAINEE_LLM_TIMEOUT_SEC`

## API

- `POST /api/project/register`
- `POST /api/project/context`
- `GET /api/project`
- `GET/POST /api/runtime/provider`
- `POST /api/loop/start`
- `POST /api/loop/stop`
- `GET /api/loop`
- `GET /api/runs`
- `GET /api/runs/{id}`
- `GET /api/events`

## 测试

```bash
pytest
```
