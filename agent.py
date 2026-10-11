# -*- coding: utf-8 -*-
"""
agent.py —— 智能体主循环（BabyAGI 核心逻辑重构）

运行流程（每轮迭代）：
  1. 记忆召回   ：从长期向量记忆召回与目标相关的高片段 + 短期工作记忆，
                  注入规划 prompt，给模型提供历史上下文；
  2. 任务规划   ：用适配 DeepSeek 的结构化系统提示词拆解子任务，
                  强制输出 JSON 数组，容错解析失败则跳过本轮；
  3. 优先级调度 ：对每个新任务计算"综合优先级"（LLM分×0.7 + 记忆分×0.3），
                  并入任务队列后按优先级降序排序，取最高优先级任务执行；
  4. 执行       ：执行任务并保存结果到分层记忆；
  5. 循环迭代   ：直到达到 max_iter 或任务队列清空。

相比原版的关键改动：
  - 生成子任务前先做记忆召回（原版无召回，直接全量拼历史）；
  - 任务去重（原版只靠 LLM 打分，易重复生成）；
  - 结构化 JSON 系统提示词 + 容错解析（适配 DeepSeek 输出习惯）。
"""
from config import MAX_ITER
from llm_client import chat_completion, extract_json
from memory import MemoryStore
from task_priority import calc_task_priority

# 任务规划：针对 DeepSeek 输出习惯重写，强约束 JSON 数组，禁止附带解释
PLAN_SYSTEM = (
    "你是任务规划智能体。根据全局目标和已完成的子任务与历史记忆，"
    "拆解出下一步需要执行的具体、可执行、与全局目标强相关的子任务。\n"
    "硬性要求：\n"
    "1. 不要重复已经列出或已经完成的子任务；\n"
    "2. 只输出严格的 JSON 数组，格式：[{\"task\": \"子任务描述\"}, ...]；\n"
    "3. 不要输出任何 JSON 之外的解释文字、Markdown 或语气词。"
)

# 任务执行：直接完成任务，输出精炼、可存储的结果
EXEC_SYSTEM = (
    "你是任务执行智能体。根据给定的任务说明直接完成任务，"
    "输出简洁、结构化的结果，突出关键信息与结论，不超过 200 字。"
)


class BabyAGI:
    def __init__(self, goal, max_iter=MAX_ITER):
        self.goal = goal
        self.max_iter = max_iter
        self.memory = MemoryStore()
        self.task_queue = []      # 待执行任务队列：[{task_id, task, priority}]
        self.task_id = 1          # 自增任务编号
        self.done_tasks = 0

    # ------------------------------------------------------------------
    # 1 + 2：记忆召回 + 任务规划
    # ------------------------------------------------------------------
    def plan_next_tasks(self):
        """召回历史记忆并让模型生成新子任务，返回去重后的任务描述列表。"""
        # 长期向量记忆（高相关片段）+ 短期工作记忆拼入 prompt
        long_docs, _ = self.memory.recall_memory(self.goal)
        short_docs = self.memory.short_term_context()

        history_block = "（无历史记忆）"
        if long_docs or short_docs:
            history_block = f"长期记忆：\n{long_docs}\n近期完成：\n{short_docs}"

        user_msg = (
            f"全局目标：{self.goal}\n"
            f"历史相关记忆：\n{history_block}\n"
            f"请生成下一步子任务。"
        )
        raw = chat_completion(PLAN_SYSTEM, user_msg)
        parsed = extract_json(raw)

        if not isinstance(parsed, list):
            print("[plan] 输出无法解析为 JSON 任务列表，跳过本轮规划")
            return []

        # 去重：与已入队任务 + 本轮内已出现的描述比对，过滤空值
        seen = set()
        fresh = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            desc = (item.get("task") or "").strip()
            if not desc or desc in seen:
                continue
            seen.add(desc)
            fresh.append(desc)
        return fresh

    # ------------------------------------------------------------------
    # 3：入队 + 优先级调度
    # ------------------------------------------------------------------
    def enqueue_tasks(self, tasks):
        for desc in tasks:
            # 记忆关联分：该任务在长期记忆里的最高相似度
            _, best_sim = self.memory.recall_memory(desc)
            priority = calc_task_priority(desc, self.goal, memory_score=best_sim)
            self.task_queue.append(
                {"task_id": self.task_id, "task": desc, "priority": priority}
            )
            self.task_id += 1
        # 按综合优先级降序，高优先级任务先执行
        self.task_queue.sort(key=lambda x: x["priority"], reverse=True)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self):
        print(f"===== 智能体启动：目标「{self.goal}」，最大迭代 {self.max_iter} 轮 =====\n")
        for i in range(1, self.max_iter + 1):
            print(f"===== 迭代轮次 {i}/{self.max_iter} =====")

            new_tasks = self.plan_next_tasks()
            self.enqueue_tasks(new_tasks)
            if new_tasks:
                print(f"[plan] 新增 {len(new_tasks)} 个子任务，当前队列 {len(self.task_queue)} 个")

            if not self.task_queue:
                print("[agent] 任务队列已空，提前结束迭代")
                break

            # 弹出最高优先级任务
            current = self.task_queue.pop(0)
            print(
                f"[exec] 执行任务 #{current['task_id']}（优先级 {current['priority']}）：{current['task']}"
            )

            result = chat_completion(
                EXEC_SYSTEM,
                f"全局目标：{self.goal}\n待执行任务：{current['task']}",
            )
            if result is None:
                result = "任务执行失败：LLM 调用异常（已兜底，不中断循环）"
            print(f"[exec] 执行结果：{result}")

            # 结果写入分层记忆
            self.memory.save_memory(current["task_id"], current["task"], result)
            self.done_tasks += 1
            print(f"[mem] 已保存长期记忆，当前长期文档 {self.memory.stats()['long_term_docs']} 条，短期记忆 {self.memory.stats()['short_term_size']} 条\n")

        print("\n===== 智能体迭代结束 =====")
        print(f"累计完成任务：{self.done_tasks} 个")
        print(f"长期向量记忆文档：{self.memory.stats()['long_term_docs']} 条")
        print(f"短期工作记忆条数：{self.memory.stats()['short_term_size']} 条")
