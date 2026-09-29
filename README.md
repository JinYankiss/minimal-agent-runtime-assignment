# Minimal Agent Runtime

这是一个不依赖 LangGraph、OpenHands、OpenClaw、PI 等 Agent 框架的最小可用 Agent。核心循环、工具注册、LLM 输出解析、Session 持久化、上下文压缩、工具超时、幂等重试和 JSONL trace 都由本项目自行实现。

## 运行环境

- Python 3.11+
- 一个 OpenAI-compatible Chat Completions API
- 真实运行需要设置 `OPENAI_API_KEY`

安装与测试：

```powershell
cd agent-runtime-assignment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

真实 LLM 运行：

```powershell
$env:PYTHONPATH = "src"
$env:OPENAI_API_KEY = "your-real-key"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_MODEL = "gpt-4o-mini"
python -m mini_agent.cli
```

也可以在项目根目录创建本地 `.env` 文件，CLI 会自动读取；`.env` 已被 `.gitignore` 忽略。真实 Key 不要写入或提交 `.env.example`：

```text
OPENAI_API_KEY=your-real-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

程序会先要求输入 `user_id` 和 `session_id`，之后进入多轮对话。工具调用日志写入 `traces/<run_id>.jsonl`，Session 默认持久化到 `sessions/sessions.json`，也可以用 `AGENT_SESSION_PATH` 覆盖。代码没有把 API Key 写入仓库，也没有提供 API 地址给用户，提交时只需提交 GitHub 仓库链接。

## 设计

```text
CLI
  -> AgentRuntime
       -> SessionStore: (user_id, session_id) 隔离并持久化上下文
       -> LLMClient: OpenAI-compatible HTTP client
       -> DecisionParser: JSON -> final/tool_call
       -> ToolRegistry: name/description/JSON Schema/effect
       -> ToolExecutor: validate/timeout/retry/idempotency ledger
       -> TraceLogger: JSONL run/tool trace
```

一次循环的状态转移是：

1. 追加用户输入到当前 Session。
2. 用 system prompt、工具 Schema、summary 和最近消息请求 LLM。
3. 解析 LLM 的 `final` 或 `tool_call` JSON。
4. 工具调用经过 Schema 校验、幂等策略判断和超时执行。
5. 把工具结果作为 `tool` 消息追加到当前 Session，继续下一轮。
6. `final` 返回答案；超过 `max_rounds` 则失败并写 trace。

如果 LLM 选择了不存在的工具，或者给出的参数不满足工具 JSON Schema，框架不会中断整个 run，而是把错误归一为 `ToolResult(ok=False)` 作为 observation 写回上下文，让下一轮 LLM 可以修正工具名、补齐参数或直接给出解释。

### 工具

内置工具：

- `calculator`：受限 AST 算术计算，不使用 `eval`。
- `search`：确定性的 mock 搜索。
- `weather`：确定性的 mock 天气。
- `todo_add`：幂等写工具，按 `(user_id, session_id)` 保存待办。
- `todo_list`：读取当前窗口待办。

每个工具都有名称、描述、JSON Schema、读写效果、超时时间和最大重试次数。LLM 不接触 Python 函数，而是基于 `ToolRegistry.schemas()` 自主选择工具。为保证不同 OpenAI-compatible 服务都遵守同一个输出协议，本实现把 Schema 放入 system prompt，并让 LLM 只返回 JSON 内容，不依赖服务商的原生 tool-call 响应格式。

### Session 与 Memory

Session 的主键是 `(user_id, session_id)`，所以同一个用户的窗口 1 和窗口 2 互不污染；同一窗口继续发送消息时，会继续使用同一个 Session。

Memory 放置方式：

- 当前短期记忆放在 `Session.messages`，包含用户输入、Agent 已执行的工具调用摘要和工具结果。
- `SessionStore` 会把 `summary` 和最近消息保存到 JSON 文件，默认路径是 `sessions/sessions.json`；该目录已加入 `.gitignore`，不会随代码提交。
- Agent 的完整思考过程不写入上下文，只允许模型输出简短、面向决策的 `thought`，避免把隐藏推理当成长期记忆。
- 消息数量超过阈值后，`SessionStore` 把较早的用户消息和工具结果压缩成有上限的 summary，并只保留最近消息。
- 每次 LLM 调用前召回：`summary + 最近 max_messages 条消息 + 当前工具 Schema`。
- 本题不实现跨 Session 的长期记忆，避免把用户 A 的窗口 1 信息错误带到窗口 2。生产实现可以另加按用户隔离的 MemoryStore，并明确召回范围。

## 幂等性与重试

详细架构回答见 [`ARCHITECTURE_IDEMPOTENCY.md`](ARCHITECTURE_IDEMPOTENCY.md)。实现中的关键点：

- `ToolEffect.READ_ONLY`：读取工具可以自动重试。
- `ToolEffect.IDEMPOTENT_WRITE`：重试使用同一个 `idempotency_key`；成功结果落入账本，重复到达直接 replay。
- `ToolEffect.NON_IDEMPOTENT_WRITE`：默认拒绝自动重试，除非接入方先把业务操作改造成带外部幂等键的接口，并注册为幂等写。
- 幂等键由 `run_id + call_id + tool_name + canonical arguments` 计算。一次 Agent run 内，重试不会因时间变化生成新键。
- timeout 不能证明远端没有执行成功，所以框架只能重试安全工具；非幂等工具必须返回人工处理或查询状态。

## 测试覆盖

测试覆盖：

- 直接回答。
- 工具调用后继续 Loop 并最终回答。
- 同一用户多个 Session 隔离。
- 上下文 summary 压缩与最近消息保留。
- 最大轮次限制。
- LLM JSON 解析失败/字段缺失。
- 工具不存在和参数校验失败会作为 observation 回灌，不中断 run。
- Session JSON 持久化和进程重建后的恢复。
- 非幂等工具不自动重试。
- 幂等写结果 replay，实际 handler 只执行一次。

## AI Prompt 与问题解决记录

见 [`PROMPT_LOG.md`](PROMPT_LOG.md)。记录只保留问题拆分、边界判断和验证过程，不把模型生成结果当成设计依据。
