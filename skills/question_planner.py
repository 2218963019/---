"""
答辩提问规划器
在答辩开始时生成逻辑递进的提问大纲，确保各轮问题连贯紧凑、层层深入。
解决"提问缺乏连贯性"问题：从松散随机提问 → 结构化递进提问。
"""

import json
import re
from typing import Optional
from skills.llm_client import LLMClient


class QuestionPlanner:
    """
    提问规划器：根据项目材料、评委人设、答辩轮数，生成一份逻辑递进的提问大纲。

    大纲示例（5轮）：
        [
            {"round": 1, "topic": "核心创新点", "goal": "切入项目最核心的差异化创新", "link": "答辩开场"},
            {"round": 2, "topic": "技术方案深度", "goal": "深挖创新点背后的技术实现", "link": "承接第1轮创新点"},
            {"round": 3, "topic": "实验验证", "goal": "考察技术方案的有效性证据", "link": "承接第2轮技术方案"},
            {"round": 4, "topic": "应用落地", "goal": "从技术转向实际应用场景", "link": "技术验证后转向应用"},
            {"round": 5, "topic": "风险与改进", "goal": "收尾考察风险意识和改进方向", "link": "全面考察后收尾"},
        ]
    """

    PLAN_PROMPT = (
        "你是一位经验丰富的答辩评审组长，需要为一场{num_turns}轮的答辩制定提问大纲。\n\n"
        "评委风格：{persona_style}\n"
        "评委关注重点：{question_focus}\n"
        "答辩场景：{scenario}\n\n"
        "项目材料摘要：\n{project_summary}\n\n"
        "请制定一份逻辑递进、层层深入的提问大纲。要求：\n"
        "- 第1轮从项目最核心的亮点切入（不要问'请介绍项目'这种泛泛问题）\n"
        "- 每一轮要承接上一轮的考察方向，形成逻辑链\n"
        "- 从具体→宏观 或 从创新→技术→验证→应用→风险 的递进结构\n"
        "- 最后一轮收尾，考察反思和改进意识\n"
        "- 每轮只聚焦一个主题，不要贪多\n\n"
        "请严格按以下JSON数组格式输出（不要输出任何其他内容，不要用markdown代码块）：\n"
        '[{{"round": 1, "topic": "主题", "goal": "本轮考察目标", "link": "与上一轮的衔接关系"}}, ...]\n'
        "注意：round从1到{num_turns}，共{num_turns}个元素。"
    )

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_plan(
        self,
        project_summary: str,
        persona_style: str,
        question_focus: str,
        scenario: str,
        num_turns: int,
    ) -> list:
        """
        生成提问大纲。

        Args:
            project_summary: 项目材料摘要
            persona_style: 评委风格描述
            question_focus: 评委关注重点
            scenario: 答辩场景
            num_turns: 提问轮数

        Returns:
            提问大纲列表 [{"round": 1, "topic": "...", "goal": "...", "link": "..."}, ...]
        """
        prompt = self.PLAN_PROMPT.format(
            num_turns=num_turns,
            persona_style=persona_style,
            question_focus=question_focus,
            scenario=scenario,
            project_summary=project_summary[:1500] if project_summary else "暂无材料",
        )

        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1024,
        )

        plan = self._parse_plan(resp.content, num_turns)
        return plan

    def _parse_plan(self, raw: str, num_turns: int) -> list:
        """从LLM输出中解析提问大纲，带兜底。"""
        text = raw.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            plan = json.loads(text[start:end])
            if isinstance(plan, list) and len(plan) > 0:
                return self._normalize_plan(plan, num_turns)
        except (ValueError, json.JSONDecodeError):
            pass

        return self._fallback_plan(num_turns)

    def _normalize_plan(self, plan: list, num_turns: int) -> list:
        """规范化大纲，确保字段完整且轮数正确。"""
        normalized = []
        for i, item in enumerate(plan[:num_turns], 1):
            if not isinstance(item, dict):
                continue
            normalized.append({
                "round": item.get("round", i),
                "topic": item.get("topic", f"第{i}轮考察"),
                "goal": item.get("goal", ""),
                "link": item.get("link", ""),
            })

        while len(normalized) < num_turns:
            idx = len(normalized) + 1
            normalized.append(self._fallback_plan(num_turns)[idx - 1])

        return normalized

    def _fallback_plan(self, num_turns: int) -> list:
        """LLM解析失败时的兜底大纲。"""
        templates = [
            ("核心创新点", "切入项目最核心的差异化创新", "答辩开场"),
            ("技术方案深度", "深挖创新点背后的技术实现细节", "承接创新点追问技术"),
            ("实验验证与数据", "考察技术方案的有效性证据", "承接技术方案追问验证"),
            ("应用落地场景", "从技术转向实际应用和用户价值", "技术验证后转向应用"),
            ("竞品对比与优势", "考察对行业现状的认知和差异化", "应用场景后追问竞争"),
            ("风险与挑战", "考察风险意识和应对方案", "全面考察后转向风险"),
            ("团队与资源", "考察执行能力和资源匹配", "风险之后考察团队"),
            ("长期规划与反思", "收尾考察战略视野和改进方向", "收尾"),
        ]
        plan = []
        for i in range(num_turns):
            t = templates[i] if i < len(templates) else templates[-1]
            plan.append({"round": i + 1, "topic": t[0], "goal": t[1], "link": t[2]})
        return plan

    def get_round_guidance(self, plan: list, round_num: int) -> dict:
        """
        获取指定轮次的提问指导。

        Returns:
            {"topic": "...", "goal": "...", "link": "...", "prev_topics": [...]}
        """
        if not plan or round_num > len(plan):
            return {"topic": "", "goal": "", "link": "", "prev_topics": []}

        current = plan[round_num - 1]
        prev_topics = [p["topic"] for p in plan[:round_num - 1]]
        return {
            "topic": current.get("topic", ""),
            "goal": current.get("goal", ""),
            "link": current.get("link", ""),
            "prev_topics": prev_topics,
        }