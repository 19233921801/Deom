# -*- coding: utf-8 -*-
"""
babyagi_deepseek.py —— 程序入口

用法：
    python babyagi_deepseek.py
    python babyagi_deepseek.py --goal "我的顶层目标" --max-iter 8

先确保已配置 .env（DEEPSEEK_API_KEY），首次运行会下载 Chroma 默认嵌入模型。
"""
import argparse

from agent import BabyAGI
from config import MAX_ITER


def main():
    parser = argparse.ArgumentParser(description="BabyAGI-DeepSeek：基于 DeepSeek 的自主智能体")
    parser.add_argument(
        "--goal",
        default="调研大模型 Agent 在工业机器视觉领域的应用现状",
        help="顶层目标，智能体将围绕它自动拆解并执行子任务",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=MAX_ITER,
        help="最大迭代轮次，防止无限循环（默认 5）",
    )
    args = parser.parse_args()

    agent = BabyAGI(goal=args.goal, max_iter=args.max_iter)
    agent.run()


if __name__ == "__main__":
    main()
