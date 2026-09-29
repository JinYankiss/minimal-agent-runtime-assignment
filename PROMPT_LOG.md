# AI Prompt 与问题解决记录

## 1. 需求拆分

先把题目拆成四条独立责任链：

1. Decision：LLM 如何表示 final/tool_call，如何校验不稳定输出。
2. Runtime：如何循环、限制轮次、回填工具结果和处理异常。
3. State：如何按用户和窗口隔离 Session，如何做基础上下文压缩。
4. Tool safety：如何注册工具契约、超时、重试和幂等账本。

这样避免把所有逻辑堆进一个 Agent 类，也避免先接框架再补业务规则。

## 2. 采用的 Prompt

Runtime 使用的系统 Prompt 约束模型只返回一个 JSON decision：

```text
You are a minimal tool-using agent.
Return exactly one JSON object and no markdown:
{"thought":"brief decision rationale","action":{"type":"final","answer":"..."}}
or
{"thought":"brief decision rationale","action":{"type":"tool_call","call_id":"stable-id","tool_name":"name","arguments":{}}}
Use a tool when it is needed. After a tool result, either call another tool or return final.
The thought must be brief and operational; do not expose private chain-of-thought.
```

工具 Schema 作为独立 system message 注入，内容包括名称、描述、参数、读写效果和重试策略。模型可以选择工具，但不能覆盖 Runtime 的安全策略。

## 3. 关键问题与决策

### LLM 输出不是工具调用协议怎么办？

不直接把任意文本交给工具层。先解析 JSON，再检查 `action.type`、`call_id`、`tool_name`、`arguments`。错误转成 `LLMProtocolError`，trace 记录原始返回，避免静默执行。

### 为什么不把完整思考过程写入 context？

题目需要看到“思考过程提取”，但 Agent 不应把隐藏推理当成长期记忆。实现只接受简短的操作理由用于 trace，不把它作为后续上下文的必要输入；后续只保留用户消息、工具调用摘要和工具结果。

### Session 怎么隔离？

用 `(user_id, session_id)` 作为内存存储 key。同一个用户的两个窗口拥有两个 Session 对象，不能只用 user_id，否则题目中的窗口 1/2 会互相污染。

后续审查发现纯内存 Session 在进程重启后无法继续对话，因此补充 JSON 持久化。`SessionStore` 负责保存和恢复 `summary + messages`，默认路径为 `sessions/sessions.json`，运行数据目录不提交到 GitHub。

### context 太长怎么处理？

达到阈值后，保留最近消息，把更早的用户消息和工具结果截断后拼成有上限 summary。每次 LLM 调用都放 summary 和最近消息；本题不做复杂的语义摘要，降低不可控依赖。

### 工具不存在或参数错误怎么办？

这类错误来自 LLM 决策不稳定，属于 Agent 可以自我修正的 observation。工具层不再把 `ToolNotFoundError` / `ToolValidationError` 直接抛出到 Runtime，而是返回 `ToolResult(ok=False)`。Runtime 按正常工具结果写回上下文，下一轮 LLM 可以更换工具、补齐参数或直接解释失败原因。

### 工具超时后是否重试？

先看工具效果类型。只读可以重试；幂等写用同一个 key 重试；非幂等写不自动重试，转状态查询。原因是 timeout 不能证明远端未执行。

## 4. 验证记录

- 使用 `unittest` 覆盖直接回答、工具循环、Session 隔离、Session 持久化恢复、上下文压缩、最大轮次、JSON 解析失败、工具契约错误 observation 回灌、非幂等禁止重试和幂等结果 replay。
- 使用 mock/scripted LLM 测试 Runtime，不让单测依赖外部 API。
- 真实 LLM 入口使用 `OpenAICompatibleClient`，密钥只从环境变量读取。
- trace 使用 JSONL，记录 run、LLM decision、tool start/finish 和 final/failure，便于面试时演示定位过程。
