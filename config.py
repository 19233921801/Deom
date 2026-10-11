# -*- coding: utf-8 -*-
"""
config.py —— 全局配置

所有可调参数集中在此处，通过 .env 或环境变量注入，便于部署与面试讲解。
"""
import os

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件（不会上传到 GitHub，见 .gitignore）
load_dotenv()

# ------------------------- DeepSeek 接口 -------------------------
# DeepSeek 官方兼容 OpenAI 的 Chat API，base_url 固定为 /v1
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

# ------------------------- LLM 调用稳定性 -------------------------
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))          # 单次请求超时（秒）
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))                   # 失败重试次数
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2.0"))           # 重试退避基数（秒）
TOKEN_LIMIT = int(os.getenv("TOKEN_LIMIT", "4000"))                # 上下文 token 上限保护
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))               # 生成温度

# ------------------------- 智能体运行控制 -------------------------
MAX_ITER = int(os.getenv("MAX_ITER", "5"))                         # 最大迭代轮次，防止无限循环

# ------------------------- 向量记忆（Chroma）----------------------
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "./memory_db")        # 向量库本地持久化路径
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "long_term_memory")
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "3"))                 # 召回条数上限
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.45"))          # 相似度阈值：低于则过滤
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))                   # 文本分块长度（字符）
SHORT_TERM_SIZE = int(os.getenv("SHORT_TERM_SIZE", "8"))           # 短期工作记忆条数上限
