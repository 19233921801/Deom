# BabyAGI-DeepSeek：基于 DeepSeek 的自主智能体重构

## 项目简介

BabyAGI 是一个极简自主智能体框架，原生依赖 OpenAI 接口实现任务规划、执行、结果存储。本项目对 BabyAGI 进行**深度改造**，不只是替换 LLM 接口，而是底层重构任务调度逻辑、分层记忆模块，将大模型切换为 DeepSeek 大模型（deepseek-chat），使用 Chroma 向量数据库实现智能体长期记忆。

智能体可以接收顶层目标，自动拆解子任务、判断任务优先级、执行任务、保存任务结果，检索历史记忆辅助后续决策，自主迭代直到完成整体目标。

## 主要改造点

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
    └── modify_log.md     # 逐条改造记录
```

## 完整运行步骤（从零到跑通）

> 以下为在 Windows / macOS / Linux 上从零运行本项目的完整流程，按顺序执行即可。

### Step 1 环境准备

1. **安装 Python 3.10+**（建议 3.10~3.12）
   - Windows：到 [python.org](https://www.python.org/downloads/) 下载安装，安装时**务必勾选 "Add Python to PATH"**；
   - 验证：`python --version` 能输出版本号即成功。
2. **准备 DeepSeek API Key**
   - 到 [platform.deepseek.com](https://platform.deepseek.com) 注册并创建 API Key；
   - Key 形如 `sk-...`，充值少量余额即可（对话模型很便宜）。

### Step 2 获取项目代码

```bash
# 方式一：clone 本仓库
git clone https://github.com/19233921801/Deom.git
cd Deom

# 方式二：对照原版 BabyAGI 学习 diff（可选）
# git clone https://github.com/yoheinakajima/babyagi.git
```

### Step 3 创建虚拟环境（推荐，隔离依赖）

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4 安装依赖

```bash
pip install -r requirements.txt
```

> 首次运行时会自动下载 Chroma 默认嵌入模型（all-MiniLM-L6-v2），需联网。

### Step 5 配置 .env

```bash
# Windows CMD
copy .env.example .env

# macOS / Linux / PowerShell
cp .env.example .env
```

打开 `.env`，填入你的 Key（其余参数保持默认即可）：

```
DEEPSEEK_API_KEY="你的deepseek api key"
DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
MODEL_NAME="deepseek-chat"
```

> ⚠️ `.env` 已列入 `.gitignore`，**不会被提交**，请勿把它传到 GitHub。

### Step 6 运行项目

```bash
# 默认目标 + 默认 5 轮迭代
python babyagi_deepseek.py

# 自定义目标与迭代轮数
python babyagi_deepseek.py --goal "调研大模型Agent在工业质检中的应用" --max-iter 8
```

### Step 7 理解运行输出（每轮迭代流程）

程序每轮迭代会依次打印：

```
===== 迭代轮次 1/5 =====
[plan] 新增 3 个子任务，当前队列 3 个      ← ① 记忆召回 + 任务拆解
[exec] 执行任务 #1（优先级 8.42）：xxx     ← ② 优先级调度 + 执行
[exec] 执行结果：xxx                       ← ③ 任务执行结果
[mem] 已保存长期记忆，当前长期文档 N 条 ...  ← ④ 写入分层记忆
```

运行结束汇总「累计完成任务数 / 长期向量记忆文档数 / 短期记忆条数」。

### Step 8 自定义与调参（可选）

所有参数集中在 `config.py`，可通过 `.env` 覆盖：

| 参数 | 作用 | 默认 |
| --- | --- | --- |
| `MAX_ITER` | 最大迭代轮次，防无限循环 | 5 |
| `SIM_THRESHOLD` | 记忆召回相似度阈值，越高召回越严 | 0.45 |
| `MEMORY_TOP_K` | 每次召回的记忆条数上限 | 3 |
| `CHUNK_SIZE` | 长期记忆文本分块长度 | 400 |
| `TOKEN_LIMIT` | 上下文 token 截断保护上限 | 4000 |
| `TEMPERATURE` | 生成温度，越低越稳定 | 0.2 |

### Step 9 重置记忆（可选）

智能体的长期记忆持久化在本地 `memory_db/` 目录。想让它"忘掉"历史、从全新状态开始：

```bash
# Windows
rmdir /s /q memory_db

# macOS / Linux
rm -rf memory_db
```

> 删除 `memory_db/` 不影响源码，下次运行会自动重建空记忆库。

### 常见问题（FAQ）

- **`ModuleNotFoundError: chromadb`** → 未安装依赖，回到 Step 4。
- **`缺少 DEEPSEEK_API_KEY`** → `.env` 未创建或 Key 为空，回到 Step 5。
- **`HTTP 401 / 402`** → API Key 错误，或 DeepSeek 账户余额不足。
- **`JSON 解析失败`** → 智能体已做兜底跳过本轮，属正常容错，可调低 `TEMPERATURE` 改善。
- **Chroma 下载嵌入模型很慢/失败** → 需稳定网络，或检查代理设置。

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

## 项目核心难点（技术解析）

- **模型适配问题**：DeepSeek 与 OpenAI 输出风格差异大，原版 prompt 会让 DeepSeek 输出大量自然语言、无法稳定输出 JSON。→ 重设计结构化提示词 + 输出格式强约束 + JSON 异常捕获。
- **记忆污染问题**：原版每次把全部历史拼进 prompt，上下文膨胀。→ 分层记忆，向量检索只召回高相关片段，相似度阈值过滤无关记忆，控制总 token。
- **任务重复生成**：原版只靠 LLM 打分，易反复生成相同任务。→ 多因子优先级加权 + 记忆相似度抑制 + 任务去重，双保险。
- **接口稳定性**：增加超时、重试、API 报错捕获，防止单次接口失败终止整个智能体循环。

## License

MIT
