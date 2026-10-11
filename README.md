# BabyAGI-DeepSeek：基于 DeepSeek 的自主智能体重构

## 项目简介

BabyAGI 是一个极简自主智能体框架，原生依赖 OpenAI 接口实现任务规划、执行、结果存储。本项目对 BabyAGI 进行**深度改造**，不只是替换 LLM 接口，而是底层重构任务调度逻辑、分层记忆模块，将大模型切换为 DeepSeek 大模型（deepseek-chat），使用 Chroma 向量数据库实现智能体长期记忆。

智能体可以接收顶层目标，自动拆解子任务、判断任务优先级、执行任务、保存任务结果，检索历史记忆辅助后续决策，自主迭代直到完成整体目标。

## 主要改造点（重点，简历 + 面试核心）

| 改造点 | 原版问题 | 本项目方案 |
| --- | --- | --- |
| **LLM 底座迁移** | 硬编码 OpenAI 调用 | 重写为统一 LLM 抽象层，适配 DeepSeek API；增加重试、超时、token 计数截断，解决上下文长度限制 |
| **任务优先级调度** | 仅靠 LLM 单次打分 | 新增优先级权重计算公式：`LLM相关性分×0.7 + 记忆关联分×0.3`，结合记忆相似度抑制重复任务 |
| **分层记忆设计** | 全部历史无筛选拼进 prompt | 拆分**短期工作记忆** + **长期向量记忆**（Chroma），召回时只注入高相关片段 |
| **向量检索优化** | 无 | 文本 chunk 分片 + 相似度阈值过滤 + 召回数量上限，减少 prompt 冗余、降低 token 消耗 |
| **提示词工程重构** | 原版 prompt 导致 DeepSeek 输出不稳定 | 重写规划/打分/执行三套结构化提示词，强制输出 JSON，容错解析 |
| **异常鲁棒性** | 单次失败即中断 | JSON 解析异常捕获、接口限流重试、记忆存储兜底，保证连续迭代 |

## 技术栈

Python 3.10+ · DeepSeek API · Chroma 向量数据库 · python-dotenv · requests · numpy

## 目录结构

```
BabyAGI-DeepSeek/
├── babyagi_deepseek.py   # 程序入口（python babyagi_deepseek.py）
├── config.py             # 全局配置（.env / 环境变量）
├── llm_client.py         # 统一 LLM 调用层（重试/超时/token截断/容错JSON）
├── memory.py             # 分层记忆：短期工作记忆 + Chroma 长期向量记忆
├── task_priority.py      # 任务优先级加权打分
├── agent.py              # 智能体主循环（记忆召回→规划→调度→执行→存记忆）
├── requirements.txt
├── .gitignore
├── .env.example
└── docs/
    └── modify_log.md     # 逐条改造记录（面试证明源码深读）
```

## 快速部署

### Step 1 环境安装

```bash
git clone https://github.com/yoheinakajima/babyagi.git   # 参考原版，便于 diff
cd babyagi_deepseek
pip install -r requirements.txt
```

> 首次运行会自动下载 Chroma 默认嵌入模型（all-MiniLM-L6-v2）。

### Step 2 配置 .env

```bash
cp .env.example .env     # 然后填入你的 key
```

```
DEEPSEEK_API_KEY="你的deepseek api key"
DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
MODEL_NAME="deepseek-chat"
```

### Step 3 运行

```bash
python babyagi_deepseek.py
python babyagi_deepseek.py --goal "你的目标" --max-iter 8
```

## 核心改造实现说明

### 1. LLM 统一调用层（`llm_client.py`）

原版使用 `openai.ChatCompletion`，本项目封装独立调用函数，用 `requests` 直连 DeepSeek，内置指数退避重试、超时与 token 截断：

```python
def chat_completion(system_prompt, user_prompt, temperature=0.2, max_tokens=None):
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    for attempt in range(MAX_RETRIES):
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 429:            # 限流指数退避
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    return None                               # 调用方兜底，不中断循环
```

### 2. 任务优先级加权（`task_priority.py`）

```python
def calc_task_priority(task_desc, goal, memory_score=0.5):
    # memory_score：该任务在长期向量记忆中的最高相似度（0~1）
    llm_score = float(deepseek打分(...))       # LLM 目标相关性 0~10
    final_score = llm_score * 0.7 + (memory_score * 10) * 0.3
    return round(final_score, 2)
```

### 3. 分层记忆（`memory.py`）

```python
# 长期向量记忆：文本分块 → 向量化落盘
collection.upsert(documents=[chunk], ids=[cid])

# 召回：阈值过滤低相关片段，只注入高相关历史
def recall_memory(query, top_k=3, threshold=0.45):
    res = collection.query(query_texts=[query], n_results=top_k)
    filtered = [doc for doc, dist in zip(res["documents"][0], res["distances"][0])
                if 1 - dist >= threshold]      # 相似度 = 1 - 距离
    return "\n".join(filtered)
```

## 项目核心难点（简历写 + 面试口述重点）

- **模型适配问题**：DeepSeek 与 OpenAI 输出风格差异大，原版 prompt 会让 DeepSeek 输出大量自然语言、无法稳定输出 JSON。→ 重设计结构化提示词 + 输出格式强约束 + JSON 异常捕获。
- **记忆污染问题**：原版每次把全部历史拼进 prompt，上下文膨胀。→ 分层记忆，向量检索只召回高相关片段，相似度阈值过滤无关记忆，控制总 token。
- **任务重复生成**：原版只靠 LLM 打分，易反复生成相同任务。→ 多因子优先级加权 + 记忆相似度抑制 + 任务去重，双保险。
- **接口稳定性**：增加超时、重试、API 报错捕获，防止单次接口失败终止整个智能体循环。

## 简历项目描述（适配机器视觉 / AI 算法岗，约 260 字）

**基于 DeepSeek 重构 BabyAGI 自主智能体 | 2025.10-2025.12**

项目基于 BabyAGI 开源框架深度改造，将底层大模型迁移至 DeepSeek-chat，并非仅替换 API 接口。重构智能体任务调度体系，设计任务优先级加权打分算法，结合 LLM 相关性评估与向量记忆相似度，对任务排序，抑制重复任务生成。搭建分层记忆架构，短期内存维护迭代上下文，基于 Chroma 向量数据库构建长期记忆；优化文本分片、相似度阈值过滤策略，动态召回相关历史信息，减少 prompt 冗余。针对 DeepSeek 模型输出特性重写多套结构化提示词，增加 JSON 格式校验、接口重试、异常捕获逻辑，提升智能体持续迭代稳定性。项目可输入顶层目标，自动拆解子任务、评估优先级、执行任务并沉淀记忆，实现自主任务循环。

## 面试问答准备

**Q：这次改造和简单替换 DeepSeek 接口最大区别是什么？**
A：简单替换只是改调用地址和 key，底层逻辑完全沿用原版。本项目重构三块核心：任务优先级打分机制、长短分层记忆、适配 DeepSeek 的结构化 prompt，解决原版任务重复、记忆无筛选、大模型输出格式不稳定的问题。

**Q：向量检索为什么要加相似度阈值？**
A：Chroma 返回 top-k 时，不管内容是否相关都会返回固定数量文档。直接全部塞进 prompt 会引入无关记忆造成记忆污染、增加 token 开销。设置阈值只保留相似度高于阈值的内容，保证召回质量。

**Q：DeepSeek 和 OpenAI 在调用时你遇到哪些差异？**
A：1. 输出习惯：DeepSeek 更容易输出自然语言说明，不严格遵守 JSON，需强化系统提示词 + 代码捕获解析异常；2. token 上限：需做上下文截断；3. 接口错误码不同：单独封装重试逻辑。

## License

MIT
