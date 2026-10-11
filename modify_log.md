# 改造记录（modify_log）

> 本文档逐条记录相对于原生 [BabyAGI](https://github.com/yoheinakajima/babyagi) 的每一处改动、动机与实现位置，面试时可证明对源码有深入阅读，而非简单换接口。

## 1. LLM 底座迁移：OpenAI → DeepSeek

- **原版**：`babyagi.py` 硬编码 `openai.ChatCompletion.create(...)`，换模型需全局改调用。
- **改动**：新增 `llm_client.py` 统一调用层，用 `requests` 直连 DeepSeek `/v1/chat/completions`；`config.py` 通过 `.env` 注入 `DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / MODEL_NAME`，切换模型只改配置。
- **新增能力**：
  - 指数退避重试（处理 429 限流与 5xx 抖动）；
  - 请求超时（`timeout=60`）；
  - 粗粒度 token 估算 + 上下文截断（`truncate_to_fit`），规避 DeepSeek 上下文长度限制；
  - 容错 JSON 解析 `extract_json`（三级兜底），应对模型在 JSON 外输出自然语言。

## 2. 任务优先级调度重构

- **原版**：`execution_agent` 每次只让 LLM 对"现有任务列表"单次打分，无记忆、无权重，打分不稳定且易重复生成相似任务。
- **改动**：新增 `task_priority.py`，多因子加权：
  ```
  综合优先级 = LLM 目标相关性分(0~10) × 0.7  +  记忆关联分(0~10) × 0.3
  ```
  其中记忆关联分来自该任务在长期向量记忆中的最高相似度。
- **配套**：`agent.py` 中对规划结果做**任务去重**（与已入队任务比对），与加权打分形成双保险，抑制重复任务。

## 3. 分层记忆设计

- **原版**：把全部历史结果拼进 `context` 传入每个 prompt，无筛选、无边界，上下文迅速膨胀（记忆污染）。
- **改动**：新增 `memory.py`：
  - **短期工作记忆** `short_term`：仅保留最近 N 轮 `(任务, 结果)`，随运行裁剪，维护本轮迭代上下文；
  - **长期向量记忆**：Chroma `PersistentClient` 落盘，任务结果**文本分块**后 `upsert` 向量化存储，跨运行保留。

## 4. 向量检索优化

- **改动**（`memory.py::recall_memory`）：
  - `n_results` 限制召回数量上限（`MEMORY_TOP_K`）；
  - 相似度阈值过滤：`similarity = 1 - distance`，低于 `SIM_THRESHOLD`（默认 0.45）的历史片段不注入 prompt；
  - 返回"高相关片段文本 + 最高相似度"，最高相似度同时供优先级计算复用。
- **效果**：减少 prompt 冗余、降低 token 消耗，保证召回质量。

## 5. 提示词工程重构

- **原版**：规划 prompt 松散，DeepSeek 会输出大量自然语言，JSON 解析常失败。
- **改动**（`agent.py`）：重写三套系统提示词——
  - `PLAN_SYSTEM`：任务规划，硬性要求"只输出 JSON 数组，禁止 JSON 外解释"；
  - `_SCORE_SYSTEM`：优先级打分，"只输出 0-10 数字"；
  - `EXEC_SYSTEM`：任务执行，"输出简洁结构化结果，≤200 字"。
- **配套**：所有解析走 `extract_json` 容错逻辑，解析失败跳过本轮而非崩溃。

## 6. 异常鲁棒性

- **改动**：
  - LLM 调用：超时 + 限流退避重试 + 缺 key 校验，失败返回 `None` 由调用方兜底；
  - JSON 解析：三级提取失败返回 `None`，规划跳过本轮；
  - 优先级打分：LLM 分解析失败用中性分 3.0 兜底；
  - 记忆存储：分块 upsert 幂等，短期记忆有界裁剪；
  - 主循环：任务队列空时提前结束，`max_iter` 限制迭代防止无限循环。

## 运行观察建议

- 可在 `agent.py` 的 `run()` 中加入每轮 `memory.stats()` 打印（已内置），观察长期文档数量与短期记忆裁剪；
- 调整 `SIM_THRESHOLD` 可直观看到召回片段数量的变化，作为"阈值过滤"这一改造点的演示。
