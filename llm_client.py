# -*- coding: utf-8 -*-
"""
llm_client.py —— 统一 LLM 调用抽象层

为什么单独抽这一层（面试核心）：
原版 BabyAGI 直接用 openai.ChatCompletion，换模型要改一堆地方。
这里把"模型底座"收敛到一个模块：
  1. 兼容 DeepSeek / OpenAI / 其他 OpenAI 兼容接口，切换只改 .env；
  2. 内置超时 + 指数退避重试，处理 429 限流与 5xx 抖动；
  3. 做粗粒度 token 估算与上下文截断，规避 DeepSeek 上下文长度限制；
  4. 提供"容错 JSON 解析"，应对模型在 JSON 外输出自然语言的常见问题。
"""
import json
import re
import time

import requests

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    MAX_RETRIES,
    MODEL_NAME,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF,
    TOKEN_LIMIT,
)


# --------------------------------------------------------------------------
# token 估算与截断（粗粒度、零依赖）
# --------------------------------------------------------------------------
def estimate_tokens(text):
    """粗略估算 token 数。中文约 1 字 ≈ 1 token，英文约 4 字符 ≈ 1 token。"""
    if not text:
        return 0
    cn = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    en = len(text) - cn
    return cn + en // 4


def truncate_to_fit(text, max_tokens=TOKEN_LIMIT, keep_head=0.6):
    """当内容超出上限时，保留头部 60% + 尾部 40%，保留核心信息同时控制 token。"""
    if estimate_tokens(text) <= max_tokens:
        return text
    head_len = int(len(text) * keep_head)
    tail = text[head_len:]
    tail_tokens = estimate_tokens(tail)
    budget = max_tokens - tail_tokens
    # 从头部起按字符截取，尽量贴合预算
    head = ""
    cur = 0
    for ch in text[:head_len]:
        if estimate_tokens(head + ch) > budget:
            break
        head += ch
        cur += 1
    return head + tail


# --------------------------------------------------------------------------
# 统一对话接口
# --------------------------------------------------------------------------
def chat_completion(system_prompt, user_prompt, temperature=0.2, max_tokens=None):
    """
    调用 DeepSeek Chat API，返回模型文本；异常返回 None。
    调用方据此做兜底，不让单次接口失败终止整个智能体循环。
    """
    if not DEEPSEEK_API_KEY:
        print("[llm] 缺少 DEEPSEEK_API_KEY，请在 .env 中配置")
        return None

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    # 上下文截断保护：超限内容先裁剪再发送
    user_prompt = truncate_to_fit(user_prompt, max_tokens=TOKEN_LIMIT)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            # 429 限流：按指数退避等待后重试
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"[llm] 触发限流(429)，{wait:.1f}s 后重试（第 {attempt + 1}/{MAX_RETRIES} 次）")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout as e:
            last_err = e
            print(f"[llm] 请求超时（第 {attempt + 1}/{MAX_RETRIES} 次）")
        except requests.exceptions.RequestException as e:
            last_err = e
            print(f"[llm] 网络/HTTP 错误：{e}")
        except (KeyError, ValueError) as e:
            last_err = e
            print(f"[llm] 响应解析失败：{e}，原始响应：{resp.text[:200] if 'resp' in dir() else 'N/A'}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF)

    print(f"[llm] 已重试 {MAX_RETRIES} 次仍失败：{last_err}")
    return None


# --------------------------------------------------------------------------
# 容错 JSON 解析
# --------------------------------------------------------------------------
def extract_json(text):
    """
    提取模型输出中的 JSON（数组或对象）。
    DeepSeek 容易在 JSON 外附带自然语言，这里做三级兜底：
      1) 整体解析；
      2) 抓取首个 [ ... ]（任务列表场景）；
      3) 抓取首个 { ... }（单对象场景）。
    全部失败返回 None，由调用方跳过本轮，避免解析崩溃。
    """
    if not text:
        return None

    # 第一级：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 第二级：抓取数组（支持嵌套括号粗略匹配）
    m = re.search(r"\[[\s\S]*\]", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 第三级：抓取对象
    m = re.search(r"\{[\s\S]*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None
