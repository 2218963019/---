"""
答辩改进报告生成器
评审结果 → 完整改进报告 + Q&A问题集(含参考答案) + 雷达图数据
"""

import json
import os
import re
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from skills.llm_client import LLMClient
from skills.rag_engine import RAGEngine
from skills.judge_engine import DefenseSession, EvaluationResult, DimensionScore, EVALUATION_CRITERIA


@dataclass
class QAItem:
    """Q&A条目"""
    question: str
    reference_answer: str
    category: str
    difficulty: str = "中"


@dataclass
class ImprovementItem:
    """改进建议条目"""
    dimension: str
    current_score: float
    target_score: float
    gap: str
    action: str
    priority: str = "中"


@dataclass
class DefenseReport:
    """答辩改进报告"""
    session_id: str = ""
    persona_name: str = ""
    scenario: str = ""
    total_score: float = 0.0
    dimension_scores: list = field(default_factory=list)
    radar_data: dict = field(default_factory=dict)
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    improvements: list = field(default_factory=list)
    qa_set: list = field(default_factory=list)
    summary: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "评委": self.persona_name,
            "场景": self.scenario,
            "总分": self.total_score,
            "各维度得分": [{"维度": s.dimension, "分数": s.score, "评语": s.comment} for s in self.dimension_scores],
            "雷达图数据": self.radar_data,
            "优势": self.strengths,
            "不足": self.weaknesses,
            "改进建议": [{"维度": i.dimension, "当前": i.current_score, "目标": i.target_score, "差距": i.gap, "行动": i.action, "优先级": i.priority} for i in self.improvements],
            "QA问题集": [{"问题": q.question, "参考答案": q.reference_answer, "类别": q.category, "难度": q.difficulty} for q in self.qa_set],
            "总结": self.summary,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# 答辩改进报告",
            f"",
            f"- **答辩ID**: {self.session_id}",
            f"- **评委类型**: {self.persona_name}",
            f"- **答辩场景**: {self.scenario}",
            f"- **总分**: {self.total_score:.1f}/100",
            f"- **生成时间**: {self.created_at}",
            f"",
            f"---",
            f"",
            f"## 各维度评分",
            f"",
        ]

        for s in self.dimension_scores:
            bar_len = 20
            filled = int(s.score / 5)
            bar = "█" * filled + "░" * (bar_len - filled)
            lines.append(f"- **{s.dimension}**: `{bar}` **{s.score:.1f}**/100")
            if s.comment:
                lines.append(f"  - {s.comment}")

        lines.extend([
            "",
            "---",
            "",
            "## 优势",
            "",
        ])
        for s in self.strengths:
            lines.append(f"- {s}")

        lines.extend([
            "",
            "## 不足",
            "",
        ])
        for w in self.weaknesses:
            lines.append(f"- {w}")

        lines.extend([
            "",
            "---",
            "",
            "## 改进建议",
            "",
        ])
        for i in self.improvements:
            lines.append(f"### [{i.priority}优先级] {i.dimension} ({i.current_score:.0f}→{i.target_score:.0f})")
            lines.append(f"- **差距**: {i.gap}")
            lines.append(f"- **行动**: {i.action}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Q&A问题集（含参考答案）",
            "",
        ])
        for idx, q in enumerate(self.qa_set, 1):
            lines.append(f"### Q{idx}. {q.question}")
            lines.append(f"- **类别**: {q.category} | **难度**: {q.difficulty}")
            lines.append(f"- **参考答案**: {q.reference_answer}")
            lines.append("")

        if self.summary:
            lines.extend([
                "---",
                "",
                "## 总结",
                "",
                self.summary,
            ])

        return "\n".join(lines)


class ReportGenerator:
    """
    答辩改进报告生成器。
    整合评审结果 + LLM生成参考答案 + 改进建议。
    """

    QA_GENERATE_PROMPT = (
        "基于以下项目材料和答辩记录，生成答辩Q&A问题集。\n\n"
        "要求：\n"
        "- 生成8-12个高频答辩问题，覆盖基础项目问题、技术细节问题、未来改进方向\n"
        "- 每个问题必须给出详细的参考答案（基于项目材料）\n"
        "- 标注问题类别(基础/技术/改进)和难度(低/中/高)\n"
        "- 参考答案要具体、有数据支撑，不要泛泛而谈\n\n"
        "项目材料：\n{context}\n\n"
        "答辩记录：\n{transcript}\n\n"
        "请严格按JSON格式输出（不要markdown代码块）：\n"
        '[{{"question": "问题", "answer": "参考答案", "category": "基础/技术/改进", "difficulty": "低/中/高"}}]'
    )

    IMPROVEMENT_PROMPT = (
        "基于评审结果，生成具体的改进建议。\n\n"
        "评审维度和分数：\n{scores}\n\n"
        "不足之处：\n{weaknesses}\n\n"
        "项目材料：\n{context}\n\n"
        "请为每个得分低于80的维度生成改进建议，严格按JSON格式输出（不要markdown代码块）：\n"
        '[{{"dimension": "维度名", "current_score": 当前分, "target_score": 目标分, "gap": "差距描述", "action": "具体行动", "priority": "高/中/低"}}]'
    )

    SUMMARY_PROMPT = (
        "请为本次答辩写一段200字以内的总结评语，包含整体表现、最大亮点和最需改进的方面。\n\n"
        "评审结果：总分{total_score}/100\n"
        "各维度：{scores_text}\n"
        "优势：{strengths}\n"
        "不足：{weaknesses}\n\n"
        "直接输出总结评语："
    )

    def __init__(self, llm_client: LLMClient, rag_engine: Optional[RAGEngine] = None):
        self.llm = llm_client
        self.rag = rag_engine

    def generate(self, session: DefenseSession) -> DefenseReport:
        """
        生成完整答辩改进报告。

        Args:
            session: 已完成评审的答辩会话

        Returns:
            DefenseReport
        """
        if not session.evaluation:
            raise RuntimeError("答辩尚未评审，请先调用judge_engine.evaluate()")

        report = DefenseReport(
            session_id=session.session_id,
            persona_name=session.persona.name if session.persona else "",
            scenario=session.scenario,
            total_score=session.evaluation.total_score,
            dimension_scores=session.evaluation.scores,
            strengths=session.evaluation.strengths,
            weaknesses=session.evaluation.weaknesses,
            created_at=datetime.now().isoformat(),
        )

        report.radar_data = self._build_radar_data(session.evaluation)
        report.improvements = self._generate_improvements(session)
        report.qa_set = self._generate_qa(session)
        report.summary = self._generate_summary(session)

        return report

    def _build_radar_data(self, evaluation: EvaluationResult) -> dict:
        """构建雷达图数据"""
        return {
            "dimensions": [s.dimension for s in evaluation.scores],
            "scores": [s.score for s in evaluation.scores],
            "max": 100,
        }

    def _get_context(self, query: str) -> str:
        if self.rag and self.rag.store.size > 0:
            return self.rag.retrieve_with_context(query, top_k=5)
        return ""

    def _build_transcript(self, session: DefenseSession) -> str:
        lines = []
        for t in session.turns:
            lines.append(f"评委: {t.question}")
            if t.answer:
                lines.append(f"学生: {t.answer}")
            for i, fq in enumerate(t.followup_questions):
                lines.append(f"追问: {fq}")
                if i < len(t.followup_answers):
                    lines.append(f"回答: {t.followup_answers[i]}")
        return "\n".join(lines)

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        brace_count = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '[':
                if start == -1:
                    start = i
            elif ch == ']':
                if start != -1:
                    return text[start:i + 1]
        brace_count = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if start == -1:
                    start = i
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0 and start != -1:
                    return text[start:i + 1]
        return text

    def _generate_qa(self, session: DefenseSession) -> list:
        """生成Q&A问题集"""
        context = self._get_context("项目全貌 创新点 技术路线")
        transcript = self._build_transcript(session)

        prompt = self.QA_GENERATE_PROMPT.format(
            context=context or session.project_summary or "暂无材料",
            transcript=transcript[:2000],
        )

        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.5)
            json_str = self._extract_json(resp.content)
            data = json.loads(json_str)
            if isinstance(data, dict):
                data = data.get("qa", data.get("questions", [data]))

            qa_set = []
            for item in data:
                qa_set.append(QAItem(
                    question=item.get("question", ""),
                    reference_answer=item.get("answer", ""),
                    category=item.get("category", "技术"),
                    difficulty=item.get("difficulty", "中"),
                ))
            return qa_set
        except (json.JSONDecodeError, ValueError, TypeError):
            return [QAItem(
                question="请介绍项目的核心创新点",
                reference_answer="请参考项目材料中的创新点章节",
                category="基础",
                difficulty="低",
            )]

    def _generate_improvements(self, session: DefenseSession) -> list:
        """生成改进建议"""
        scores_text = "\n".join(
            f"- {s.dimension}: {s.score:.1f}/100"
            for s in session.evaluation.scores
        )
        weaknesses = "\n".join(f"- {w}" for w in session.evaluation.weaknesses) if session.evaluation.weaknesses else "无明显不足"
        context = self._get_context("改进方向 优化方案")

        prompt = self.IMPROVEMENT_PROMPT.format(
            scores=scores_text,
            weaknesses=weaknesses,
            context=context or session.project_summary or "暂无材料",
        )

        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.4)
            json_str = self._extract_json(resp.content)
            data = json.loads(json_str)
            if isinstance(data, dict):
                data = data.get("improvements", [data])

            improvements = []
            for item in data:
                improvements.append(ImprovementItem(
                    dimension=item.get("dimension", ""),
                    current_score=float(item.get("current_score", 0)),
                    target_score=float(item.get("target_score", 80)),
                    gap=item.get("gap", ""),
                    action=item.get("action", ""),
                    priority=item.get("priority", "中"),
                ))
            return improvements
        except (json.JSONDecodeError, ValueError, TypeError):
            improvements = []
            for s in session.evaluation.scores:
                if s.score < 80:
                    improvements.append(ImprovementItem(
                        dimension=s.dimension,
                        current_score=s.score,
                        target_score=80.0,
                        gap=f"{s.dimension}得分低于80分",
                        action=f"重点提升{ s.dimension}方面的表现",
                        priority="高" if s.score < 60 else "中",
                    ))
            return improvements

    def _generate_summary(self, session: DefenseSession) -> str:
        """生成总结评语"""
        scores_text = ", ".join(f"{s.dimension}{s.score:.0f}" for s in session.evaluation.scores)
        strengths = "; ".join(session.evaluation.strengths) if session.evaluation.strengths else "无"
        weaknesses = "; ".join(session.evaluation.weaknesses) if session.evaluation.weaknesses else "无"

        prompt = self.SUMMARY_PROMPT.format(
            total_score=session.evaluation.total_score,
            scores_text=scores_text,
            strengths=strengths,
            weaknesses=weaknesses,
        )

        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.5, max_tokens=300)
            return resp.content.strip()
        except Exception:
            return f"答辩总分{session.evaluation.total_score:.1f}/100，建议重点改进低分维度。"

    def export_report(self, report: DefenseReport, output_dir: str) -> dict:
        """
        导出报告到目录，生成JSON + Markdown两种格式。

        Args:
            report: 答辩报告
            output_dir: 输出目录

        Returns:
            生成的文件路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        base = f"report_{report.session_id}"

        json_path = os.path.join(output_dir, f"{base}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        md_path = os.path.join(output_dir, f"{base}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())

        return {"json": json_path, "markdown": md_path}