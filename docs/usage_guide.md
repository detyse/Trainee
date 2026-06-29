# Trainee 使用说明

本文档用于沉淀 Trainee 的日常使用方法，并作为本 session 后续对话的汇总位置。后续如果有新的配置约定、运行步骤、问题排查或实现决策，将追加到“Session 汇总”中。

## 项目定位

Trainee 是一个用于自动化外部模型训练循环的 agent runtime。它不包含训练代码，而是连接到已有训练项目，执行项目自己的训练命令，监控训练进度，解析指标，并根据已配置且可用的 LLM 决定下一轮参数。

典型流程：

```text
读取项目上下文 -> 运行训练 -> 监控 activity -> 解析指标
-> 对比 baseline / best-so-far -> 决定下一组参数 -> 继续或停止
```

## 安装与验证

推荐用 `uv` 从本仓库安装：

```bash
uv tool install --editable . --force
uv tool update-shell
trainee version
trainee --help
```

本地开发时也可以不全局安装：

```bash
uv run trainee --help
```

## 初始化训练项目

进入需要被 Trainee 控制的训练项目目录：

```bash
cd /path/to/training-project
trainee init
```

如有 baseline 配置文件：

```bash
trainee init --baseline-config configs/base.yaml
```

初始化后重点检查这些文件：

- `.trainee/project.yaml`: 主运行配置。
- `.trainee/tuning.yaml`: 允许 Trainee 调整的参数白名单。
- `.trainee/context.md`: Trainee 对项目的上下文理解。
- `.trainee/README.md`: 初始化时检测到的候选入口、配置和目录说明。

`trainee init` 默认会对当前 provider 做一次 live API 测试。离线初始化时可以使用 `--skip-provider-test`，但真正运行训练循环时不能跳过 provider 检查。

## 准备与校验

如果设置了 `launch.baseline_config`，可以运行：

```bash
trainee prepare
```

`prepare` 会基于项目上下文和 baseline 配置推断输出路径与可调参数。推断结果必须人工检查后再运行训练。

`trainee prepare` 同样会测试 provider；离线配置阶段可加 `--skip-provider-test`。

运行前建议校验配置并查看最终命令：

```bash
trainee doctor
trainee run --dry-run
```

## 运行训练循环

确认配置无误后启动：

```bash
trainee run
```

默认推荐使用 guarded 模式。若系统没有 `bubblewrap`，或者训练任务必须写入 `.trainee/` 之外的位置，需要显式使用不安全模式：

```bash
trainee run --unsafe
```

## Web UI

启动本地 Web UI：

```bash
trainee webui
```

默认访问：

```text
http://127.0.0.1:8000
```

Web UI 可用于项目设置、循环控制、prompt 预览、运行检查、provider 设置和报告查看。

Web UI 的 runtime health 只表示服务和数据库健康；LLM 鉴权需要使用 Provider Settings 里的 provider live test 或 `/llm-test`。

## 配置要点

`.trainee/project.yaml` 和 `.trainee/tuning.yaml` 是 CLI、Web UI 与 tool API 的共同配置来源。

常见重点：

- `launch.environment`: 运行环境，支持 `system`、`uv`、`venv`、`conda`。
- `launch.command`: 实际训练入口命令。
- `launch.baseline_config`: baseline 配置文件路径。
- `output.config_path`: baseline 配置里控制训练输出目录的字段名，例如 `output.root`。
- `run.max_rounds`: 最大训练轮数。
- `run.timeout_minutes`: 单轮超时时间。
- `metrics.specs`: 指标解析规则。
- `advanced.security_mode`: `guarded` 或 unsafe 相关配置。
- `advanced.signal_sources`: activity monitor 的观测来源，只用于 UI 状态，不参与失败判断。

运行中唯一会硬终止训练进程的是 `run.timeout_minutes`。Activity monitor 只更新 UI，不会 kill 进程，也不会把 round 标记为失败。

可调参数放在 `.trainee/tuning.yaml` 的 `params` 中。对于 config-backed 参数，初始值来自 `launch.baseline_config`，不需要在 `tuning.yaml` 中重复写 baseline 默认值。

`trainee init` 生成的 `.trainee/project.yaml` 会预留：

```yaml
output:
  config_path:
```

这里填的是字段名，不是输出目录路径。例如 baseline config 里是 `output.root: outputs`，就写：

```yaml
output:
  config_path: output.root
```

Trainee 每轮会把该字段改成当前 round 的内部输出目录；这个目录值不需要在 `project.yaml` 里另配。

## 使用案例：conda ANSR 环境

如果训练项目需要在 conda 的 `ANSR` 环境中运行，`.trainee/project.yaml` 里应在 `launch` 下这样写：

```yaml
launch:
  environment: conda
  env_name: ANSR
  command:
    - python
    - main.py
```

其中 `environment` 固定写 `conda`，真实的 conda 环境名写在 `env_name`。上面的配置会让 Trainee 以类似下面的方式启动训练：

```bash
conda run -n ANSR python main.py
```

如果训练入口不是 `main.py`，只需要把 `command` 改成项目实际入口，例如：

```yaml
command:
  - python
  - train.py
```

## Session 汇总

### 2026-06-27

- 创建本文档作为本 session 后续对话的使用说明汇总位置。
- 当前约定：后续如果对 Trainee 的使用方法、配置方式、命令、排查步骤或实现决策有新增结论，将同步追加到本节。
- 使用案例：conda 环境名为 `ANSR` 时，`launch.environment` 写 `conda`，`launch.env_name` 写 `ANSR`。
