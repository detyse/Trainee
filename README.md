# Trainee

Trainee 是一个面向训练自动化的 agent runtime。它不承载训练代码本身，而是接入外部训练项目与外部训练环境，围绕“读上下文 -> 发起训练 -> 监控 heartbeat -> 解析结果 -> 决定下一轮参数 -> 自动继续或停止”这条闭环，提供一个最小可用的 runtime 和控制台。

## 能力范围

- 注册一个本机外部训练项目，并保存项目根目录、工作目录、启动命令模板、数据路径、日志路径、可调参数和指标规则。
- 自动扫描外部项目，生成 `ProjectContext`，把“项目是什么、数据在哪、训练入口在哪、参数怎么调、结果怎么看”固化进上下文。
- 使用显式 launcher 执行训练，例如 `conda run ... python train.py ...` 或 `/path/to/train.sh ...`。
- 通过本地子进程输出和日志更新时间做 heartbeat 检测；超过阈值则标记 stalled。
- 解析日志里的 `loss/total_loss`，并支持按配置从正则或 W&B summary 中抽取更多指标。
- 通过轻量控制台查看项目信息、loop 状态、轮次历史、命令、参数、日志路径、W&B 链接和 agent 决策。

## 快速开始

1. 安装依赖：

```bash
python3 -m pip install -e ".[dev]"
```

2. 可选：从示例配置生成 `.env`：

```bash
cp .env.example .env
```

3. 启动服务：

```bash
python3 main.py
```

4. 打开 `http://127.0.0.1:8000`

5. 填写外部训练项目配置。`launcher_template` 可以直接写完整命令，例如：

```bash
conda run -n trainer python {project_root}/train.py --config {project_root}/configs/base.yaml {extra_args}
```

支持的模板变量：

- `{project_root}`
- `{working_dir}`
- `{extra_args}`

`{extra_args}` 由 runtime 根据 `tunable_params` 自动拼成 CLI 参数。agent 不会自由生成新的 shell 片段，只能覆盖白名单参数。

## 配置字段

- `project_root`: 外部训练项目根目录
- `working_dir`: 训练命令实际执行目录
- `launcher_template`: 完整启动命令模板
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

## `.env` 与 LLM Provider

启动时会自动读取仓库根目录下的 `.env`。未配置 LLM key 时，runtime 会自动回退到启发式调参。

OpenAI-compatible 示例：

```bash
TRAINEE_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
TRAINEE_LLM_TIMEOUT_SEC=30
```

Claude / Anthropic 示例：

```bash
TRAINEE_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
ANTHROPIC_VERSION=2023-06-01
ANTHROPIC_MAX_TOKENS=1024
TRAINEE_LLM_TIMEOUT_SEC=30
```

如果不显式设置 `TRAINEE_LLM_PROVIDER`，runtime 会自动优先使用已存在的 key：

- 有 `OPENAI_API_KEY` 时走 `openai`
- 只有 `ANTHROPIC_API_KEY` 时走 `anthropic`
- 两者都没有时禁用 LLM，回退到启发式策略

## LLM 决策

- `openai` provider 调用 `OPENAI_BASE_URL/chat/completions`，使用 `OPENAI_MODEL` 生成结构化 `AgentDecision`。
- `anthropic` provider 调用 `ANTHROPIC_BASE_URL/v1/messages`，使用 Claude Messages API 生成结构化 `AgentDecision`。
- 如果没有配置 LLM，runtime 会回退到一个保守的启发式策略，优先调整 `lr` / `learning_rate` 一类参数。

相关环境变量：

- `TRAINEE_DATA_DIR`
- `TRAINEE_LLM_PROVIDER`
- `LLM_PROVIDER`
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
