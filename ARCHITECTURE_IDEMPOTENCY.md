# 第 4 题：Agent 的幂等性与重试设计

## 1. 问题定义

Agent 不是一次 LLM 调用，而是：LLM 决策 -> 工具调用 -> 工具结果回填 -> LLM 再决策。任意一层都可能超时或返回不确定结果：

- LLM 请求超时，无法确定模型是否已经生成了决策。
- 工具 HTTP 超时，无法确定远端是否已经写库或扣款。
- 进程在工具成功后、写入本地状态前崩溃。
- LLM 重复输出相同工具调用。

如果只按“失败就重试”，订单、扣款、发券等副作用会被执行多次。

## 2. 工具契约

工具注册时必须声明副作用类型，而不是让 Runtime 猜：

```text
READ_ONLY
    查询、搜索、计算。默认允许有限次自动重试。

IDEMPOTENT_WRITE
    写操作必须接受 idempotency_key，并由业务服务或数据库唯一约束去重。
    Runtime 可以用同一个 key 重试。

NON_IDEMPOTENT_WRITE
    例如没有幂等键的扣款、发送短信、创建外部订单。
    Runtime 默认不自动重试，只返回“结果未知，需查询状态或人工处理”。
```

`ToolSpec` 同时声明 `timeout_seconds`、`max_retries` 和 `effect`，Runtime 据此选择策略。LLM 只能选择工具，不能覆盖这个安全策略。

## 3. 幂等键

本项目使用：

```text
SHA256(run_id + call_id + tool_name + canonical_json(arguments))
```

其中 JSON 使用排序后的 key 和固定分隔符，保证相同参数生成相同结果。重试期间不重新生成 key。

生产系统建议把 key 设计成业务可追踪的形式，例如：

```text
agent:{tenant_id}:{session_id}:{run_id}:{call_id}
```

并在业务服务中建立唯一索引：

```sql
UNIQUE (tenant_id, idempotency_key)
```

相同 key 的请求返回第一次成功结果，而不是再次执行业务写入。

## 4. Runtime 执行流程

```text
收到 tool_call
    |
    v
根据 tool_name 读取 ToolSpec，校验 JSON Schema
    |
    v
计算稳定 idempotency_key，查询 ledger
    |-- 已 completed --> replay 第一次结果
    |
    |-- NON_IDEMPOTENT_WRITE --> 禁止自动重试，返回结果未知
    |
    v
执行，设置 timeout
    |-- 成功 --> 写入 completed(key, result)
    |-- 失败且可安全重试 --> 使用同一 key 重试
    |-- 最终失败 --> 写入 failed，返回结构化工具错误
    |-- timeout --> 按 effect 分类，不把 timeout 当成“肯定没执行”
```

本项目的内存 `ToolExecutionLedger` 是演示实现。生产环境必须换成 Redis、数据库或具备持久化能力的幂等服务，并使用原子 `reserve/complete`，否则进程重启后会丢失去重记录。

## 5. timeout 的边界

timeout 只表示调用方没有在期限内拿到结果，不表示远端没有执行。对于幂等写，重试请求仍携带同一 key，由下游去重；对于非幂等写，不能直接重试，应该：

1. 用原业务单号或 request id 查询执行状态。
2. 查询到成功就把成功结果回填 Agent。
3. 查询到失败且确认未产生副作用，才允许重新发起。
4. 查询不到状态时转人工或等待异步回调，不盲目重试。

这也是为什么“工具是幂等的”必须成为注册契约，不能由 LLM 自己声称安全。

## 6. Agent 感知方式

Agent 不直接决定是否重试。它只收到结构化结果：

```json
{
  "ok": false,
  "error": "timeout: result unknown",
  "retryable": false,
  "requires_status_query": true
}
```

Runtime 根据 `ToolSpec.effect` 生成这个结果，并把 `replayed`、`attempts` 写入 trace。这样模型可以继续执行“查询状态”工具，但不能绕开 Runtime 重新调用危险工具。

## 7. 并发与一致性

同一个幂等 key 可能同时到达两个 Worker，单机字典不够。生产实现要保证：

- `SETNX` 或数据库唯一键抢占 key。
- `running` 状态有租约和过期时间，避免 Worker 崩溃后永久阻塞。
- `completed` 保存可重放的序列化结果。
- 处理中重复请求可等待、轮询或返回处理中，不重新执行业务。
- ledger 的 TTL 覆盖最大重试窗口、消息重投窗口和业务补偿窗口。

## 8. 结论

重试安全性的核心不是“多试几次”，而是“副作用操作有稳定业务键、下游有唯一约束、Runtime 按工具契约分类处理未知结果”。读取工具可以重试，幂等写必须同 key 重试，非幂等写不能自动重试；timeout 必须走状态查询或人工补偿。
