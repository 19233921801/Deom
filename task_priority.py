# -*- coding: utf-8 -*-
"""
task_priority.py —— 任务优先级调度（面试核心之一）

原版 BabyAGI 只靠 LLM 单次打分决定下一个任务，打分不稳定、容易重复生成相似任务。
本模块引入**多因子加权**：
    综合优先级 = LLM 目标相关性分(0~10) × 0.7  +  记忆关联分(0~10) × 0.3

其中：
  - LLM 分：评估当前任务与全局目标的相关性；
  - 记忆分：当前任务描述在长期记忆库中的最高相似度（0~1）放大到 0~10 区间。
    任务越"像"已经做过的内容，记忆分越高——既避免重复去做已解决的事，
    又避免同质任务反复生成时全排前面。

在 agent 层还会叠加"重复任务过滤"，与这里的加权打分配合，双保险抑制重复。
"""
from llm_client import chat_completion

# 打分专用系统提示词：只返回数字，禁止多余输出
_SCORE_SYSTEM = (
    "你是任务优先级评估器。评估给定任务与全局目标的相关性。"
    "只输出一个 0-10 之间的数字（0 表示完全无关，10 表示高度相关），"
    "不要输出任何解释或其它文字。"
)

# 加权系数
LLM_WEIGHT = 0.7   # 目标相关性权重
MEM_WEIGHT = 0.3   # 记忆关联权重


def calc_task_priority(task_desc, goal, memory_score=0.5):
    """
    计算综合优先级，返回 0~10 的分数。
    memory_score：向量检索得到的相似度（0~1），来自 memory.recall_memory。
    """
    user_prompt = f"全局目标：{goal}\n待评估任务：{task_desc}"
    score_str = chat_completion(_SCORE_SYSTEM, user_prompt, temperature=0.1)

    # 解析 LLM 打分；解析失败用中性分 3.0 兜底，不让异常中断调度
    try:
        llm_score = float(score_str.strip())
        llm_score = max(0.0, min(10.0, llm_score))
    except (AttributeError, ValueError, TypeError):
        llm_score = 3.0

    # 记忆分放大到 0~10 后加权
    mem_score = max(0.0, min(1.0, memory_score)) * 10.0
    final_score = llm_score * LLM_WEIGHT + mem_score * MEM_WEIGHT
    return round(final_score, 2)
