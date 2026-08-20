"""
教师端助手Skill
批量项目材料智能解析 → 评审标准匹配 → 问题集生成(含参考答案) → 导出
"""

import json
import os
import re
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from skills.llm_client import LLMClient
from skills.rag_engine import RAGEngine
from skills.doc_parser import DocParserSkill
from skills.question_generator import QuestionGeneratorSkill
from skills.solution_optimizer import SolutionOptimizerSkill
from skills.judge_engine import EVALUATION_CRITERIA


@dataclass
class ProjectAnalysis:
    """单个项目材料分析结果"""
    filename: str
    text: str = ""
    char_count: int = 0
    sections: dict = field(default_factory=dict)
    keywords_matched: list = field(default_factory=list)
    criteria_scores: dict = field(default_factory=dict)
    summary: str = ""


@dataclass
class QuestionSetItem:
    """问题集条目"""
    question: str
    reference_answer: str
    category: str
    difficulty: str = "中"
    scoring_points: list = field(default_factory=list)
    source_section: str = ""


@dataclass
class TeacherReport:
    """教师端报告"""
    project_analyses: list = field(default_factory=list)
    question_set: list = field(default_factory=list)
    criteria_matching: dict = field(default_factory=dict)
    common_weaknesses: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "项目分析": [
                {
                    "文件名": a.filename,
                    "字数": a.char_count,
                    "章节": list(a.sections.keys()),
                    "匹配关键词": a.keywords_matched,
                    "评审维度得分": a.criteria_scores,
                    "摘要": a.summary,
                }
                for a in self.project_analyses
            ],
            "问题集": [
                {
                    "问题": q.question,
                    "参考答案": q.reference_answer,
                    "类别": q.category,
                    "难度": q.difficulty,
                    "评分要点": q.scoring_points,
                    "来源章节": q.source_section,
                }
                for q in self.question_set
            ],
            "评审标准匹配": self.criteria_matching,
            "共性薄弱项": self.common_weaknesses,
            "教学建议": self.suggestions,
        }

    def to_markdown(self) -> str:
        lines = ["# 教师端分析报告\n"]

        for a in self.project_analyses:
            lines.append(f"## {a.filename}\n")
            lines.append(f"- 字数: {a.char_count}")
            lines.append(f"- 章节: {', '.join(a.sections.keys())}")
            lines.append(f"- 摘要: {a.summary}\n")
            if a.criteria_scores:
                lines.append("**评审维度预评估:**")
                for dim, score in a.criteria_scores.items():
                    lines.append(f"- {dim}: {score}/100")

        lines.append("\n---\n## 答辩问题集\n")
        for idx, q in enumerate(self.question_set, 1):
            diff_mark = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(q.difficulty, "")
            lines.append(f"### Q{idx}. {q.question}")
            lines.append(f"- 类别: {q.category} | 难度: {diff_mark}{q.difficulty} | 来源: {q.source_section}")
            lines.append(f"- **参考答案**: {q.reference_answer}")
            if q.scoring_points:
                lines.append(f"- **评分要点**: {'; '.join(q.scoring_points)}")
            lines.append("")

        if self.common_weaknesses:
            lines.append("---\n## 共性薄弱项\n")
            for w in self.common_weaknesses:
                lines.append(f"- {w}")

        if self.suggestions:
            lines.append("\n## 教学建议\n")
            for s in self.suggestions:
                lines.append(f"- {s}")

        return "\n".join(lines)


class TeacherAssistant:
    """
    教师端助手：批量解析项目材料 → 评审标准匹配 → 生成问题集
    """

    ANALYZE_PROMPT = (
        "请分析以下项目材料，输出：\n"
        "1. 项目摘要（100字以内）\n"
        "2. 各评审维度的预估得分（0-100），基于以下标准：\n"
        "{criteria}\n\n"
        "项目材料：\n{content}\n\n"
        "严格按JSON格式输出（不要markdown代码块）：\n"
        '{{"summary": "摘要", "scores": {{"维度1": 分数, ...}}}}'
    )

    QUESTION_SET_PROMPT = (
        "你是一位资深答辩评委，正在为以下项目准备答辩问题集。\n\n"
        "【核心要求】每个问题必须针对项目中提到的具体内容提问，禁止泛泛而谈！\n"
        "- 错误示例：'请介绍你的创新点' → 太泛\n"
        "- 正确示例：'你提到使用CLIP模型进行图文对齐，请说明CLIP的视觉编码器输出维度是多少，如何与768维的图谱嵌入对齐？' → 针对具体技术细节\n\n"
        "【生成规则】\n"
        "1. 基础项目问题(4-5个)：针对项目背景中提到的具体问题/痛点、研究目标中的具体指标、创新点的具体方法名提问\n"
        "2. 技术细节问题(5-6个)：针对技术路线中提到的具体模型/算法/框架，追问原理、参数、对比、消融实验数据\n"
        "3. 改进方向问题(3-4个)：针对局限性中提到的具体短板，追问具体改进方案和预期效果\n"
        "4. 每个参考答案必须引用材料中的具体数据/方法名/实验结果\n"
        "5. 评分要点要具体到'答出XX方法得X分'这种粒度\n\n"
        "评审标准：\n{criteria}\n\n"
        "项目材料：\n{content}\n\n"
        "严格按JSON数组格式输出（不要markdown代码块，不要在JSON外加任何文字）：\n"
        '[{{"question": "针对具体内容的提问", "answer": "基于材料的具体参考答案", "category": "基础/技术/改进", "difficulty": "低/中/高", "scoring_points": ["答出XX得X分", ...], "source_section": "来源章节"}}]'
    )

    TOPIC_EXTRACT_PROMPT = (
        "从以下项目材料中提取答辩关键考点（评委最可能追问的具体技术点）。\n\n"
        "要求：\n"
        "- 每个考点必须是材料中明确提到的具体内容（方法名、数据、指标、框架名等）\n"
        "- 不要提取泛泛的考点（如'创新性'），要提取具体的（如'CLIP与GNN的对齐方式'）\n"
        "- 每个考点附带材料中相关的原文片段\n\n"
        "项目材料：\n{content}\n\n"
        "严格按JSON数组格式输出（不要markdown代码块）：\n"
        '[{{"topic": "具体考点", "context": "材料中相关原文", "category": "基础/技术/改进", "difficulty": "低/中/高"}}]'
    )

    TARGETED_QUESTION_PROMPT = (
        "你是一位资深答辩评委，请针对以下考点生成1-2个深入的答辩问题。\n\n"
        "【严格要求】\n"
        "- 问题必须针对考点中的具体内容，追问原理/细节/数据/对比\n"
        "- 参考答案必须基于材料原文，引用具体数据和方法名\n"
        "- 评分要点要具体到'答出XX得X分'\n\n"
        "考点：{topic}\n"
        "相关材料：{context}\n"
        "类别：{category}\n\n"
        "严格按JSON数组格式输出（不要markdown代码块）：\n"
        '[{{"question": "针对考点的问题", "answer": "基于材料的参考答案", "category": "{category}", "difficulty": "{difficulty}", "scoring_points": ["答出XX得X分", ...], "source_section": "来源"}}]'
    )

    WEAKNESS_PROMPT = (
        "基于以下多个项目的分析结果，找出共性薄弱项和教学建议。\n\n"
        "各项目评审得分：\n{scores_summary}\n\n"
        "各项目摘要：\n{summaries}\n\n"
        "严格按JSON格式输出（不要markdown代码块）：\n"
        '{{"common_weaknesses": ["薄弱项1", ...], "suggestions": ["建议1", ...]}}'
    )

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.parser = DocParserSkill()
        self.rag = RAGEngine(chunk_size=400)
        self.qgen = QuestionGeneratorSkill()
        self.optimizer = SolutionOptimizerSkill()

    def analyze_document(self, file_path: str, scenario: str = "大创立项") -> ProjectAnalysis:
        """
        解析单个项目文档并分析。

        Args:
            file_path: 文档路径
            scenario: 答辩场景

        Returns:
            ProjectAnalysis
        """
        filename = os.path.basename(file_path)
        text = self.parser.parse(file_path)
        sections = self.parser.get_sections(keywords=[
            "项目背景", "研究目标", "创新点", "技术路线", "算法",
            "实验", "系统架构", "局限性", "未来规划",
            "摘要", "方法", "结果", "结论",
        ])

        analysis = ProjectAnalysis(
            filename=filename,
            text=text,
            char_count=len(text),
            sections={k: v[:200] for k, v in sections.items()},
        )

        keywords = ["知识图谱", "多模态", "视觉", "深度学习", "NLP", "算法", "创新", "实验", "系统", "应用"]
        analysis.keywords_matched = [kw for kw in keywords if kw in text]

        criteria = EVALUATION_CRITERIA.get(scenario, EVALUATION_CRITERIA["大创立项"])
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in criteria.items())

        prompt = self.ANALYZE_PROMPT.format(criteria=criteria_text, content=text[:3000])
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
            json_str = self._extract_json(resp.content)
            data = json.loads(json_str)
            analysis.summary = data.get("summary", "")
            analysis.criteria_scores = {k: float(v) for k, v in data.get("scores", {}).items()}
        except Exception:
            analysis.summary = text[:100] + "..."

        return analysis

    def analyze_text(self, text: str, source: str = "直接输入", scenario: str = "大创立项") -> ProjectAnalysis:
        """解析纯文本"""
        analysis = ProjectAnalysis(filename=source, text=text, char_count=len(text))
        criteria = EVALUATION_CRITERIA.get(scenario, EVALUATION_CRITERIA["大创立项"])
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
        prompt = self.ANALYZE_PROMPT.format(criteria=criteria_text, content=text[:3000])
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
            json_str = self._extract_json(resp.content)
            data = json.loads(json_str)
            analysis.summary = data.get("summary", "")
            analysis.criteria_scores = {k: float(v) for k, v in data.get("scores", {}).items()}
        except Exception:
            analysis.summary = text[:100] + "..."
        return analysis

    def generate_question_set(
        self,
        project_text: str,
        scenario: str = "大创立项",
        num_questions: int = 12,
    ) -> list:
        """
        基于项目材料生成针对性问题集（两步法：提取考点→逐考点深挖）。
        """
        criteria = EVALUATION_CRITERIA.get(scenario, EVALUATION_CRITERIA["大创立项"])
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in criteria.items())

        # 第一步：提取关键考点
        topics = self._extract_topics(project_text)

        # 第二步：逐考点生成针对性问题
        if topics:
            items = self._generate_targeted_questions(topics, num_questions)
            if items:
                return items

        # 兜底：一次性生成
        prompt = self.QUESTION_SET_PROMPT.format(criteria=criteria_text, content=project_text[:4000])
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.5, max_tokens=4096)
            json_str = self._extract_json(resp.content)
            data = json.loads(json_str)
            if isinstance(data, dict):
                data = data.get("questions", data.get("question_set", [data]))
            if not isinstance(data, list):
                data = [data]
            items = []
            for item in data[:num_questions]:
                items.append(QuestionSetItem(
                    question=item.get("question", ""),
                    reference_answer=item.get("answer", ""),
                    category=item.get("category", "技术"),
                    difficulty=item.get("difficulty", "中"),
                    scoring_points=item.get("scoring_points", []),
                    source_section=item.get("source_section", ""),
                ))
            if items:
                return items
        except Exception as e:
            print(f"[TeacherAssistant] 一次性生成失败: {e}")

        return self._fallback_question_set(project_text)

    def _extract_topics(self, project_text: str) -> list:
        """从材料中提取答辩关键考点"""
        prompt = self.TOPIC_EXTRACT_PROMPT.format(content=project_text[:4000])
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=2048)
            json_str = self._extract_json(resp.content)
            data = json.loads(json_str)
            if isinstance(data, dict):
                data = data.get("topics", [data])
            if not isinstance(data, list):
                data = [data]
            return [t for t in data if t.get("topic")]
        except Exception as e:
            print(f"[TeacherAssistant] 考点提取失败: {e}")
            return []

    def _generate_targeted_questions(self, topics: list, num_questions: int) -> list:
        """逐考点生成针对性问题"""
        all_items = []
        for topic in topics:
            if len(all_items) >= num_questions:
                break
            prompt = self.TARGETED_QUESTION_PROMPT.format(
                topic=topic.get("topic", ""),
                context=topic.get("context", "")[:1000],
                category=topic.get("category", "技术"),
                difficulty=topic.get("difficulty", "中"),
            )
            try:
                resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.4, max_tokens=1024)
                json_str = self._extract_json(resp.content)
                data = json.loads(json_str)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    if item.get("question"):
                        all_items.append(QuestionSetItem(
                            question=item.get("question", ""),
                            reference_answer=item.get("answer", ""),
                            category=item.get("category", topic.get("category", "技术")),
                            difficulty=item.get("difficulty", topic.get("difficulty", "中")),
                            scoring_points=item.get("scoring_points", []),
                            source_section=item.get("source_section", ""),
                        ))
            except Exception:
                continue
        return all_items


    def batch_analyze(self, file_paths: list, scenario: str = "大创立项") -> TeacherReport:
        """
        批量分析多个项目文档，生成综合报告。

        Args:
            file_paths: 文档路径列表
            scenario: 答辩场景

        Returns:
            TeacherReport
        """
        report = TeacherReport(created_at=datetime.now().isoformat())

        all_texts = []
        for fp in file_paths:
            try:
                analysis = self.analyze_document(fp, scenario)
                report.project_analyses.append(analysis)
                all_texts.append(analysis.text)
            except Exception as e:
                report.project_analyses.append(ProjectAnalysis(
                    filename=os.path.basename(fp),
                    summary=f"解析失败: {e}",
                ))

        if all_texts:
            combined = "\n\n---\n\n".join(all_texts)
            report.question_set = self.generate_question_set(combined, scenario)

        if len(report.project_analyses) > 1:
            self._analyze_common(report, scenario)

        return report

    def _analyze_common(self, report: TeacherReport, scenario: str) -> None:
        """分析多个项目的共性薄弱项"""
        scores_summary = []
        summaries = []
        for a in report.project_analyses:
            if a.criteria_scores:
                scores_summary.append(f"{a.filename}: {a.criteria_scores}")
            if a.summary:
                summaries.append(f"{a.filename}: {a.summary}")

        prompt = self.WEAKNESS_PROMPT.format(
            scores_summary="\n".join(scores_summary),
            summaries="\n".join(summaries),
        )

        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.4)
            json_str = self._extract_json(resp.content)
            data = json.loads(json_str)
            report.common_weaknesses = data.get("common_weaknesses", [])
            report.suggestions = data.get("suggestions", [])
        except Exception:
            pass

    def export_report(self, report: TeacherReport, output_dir: str) -> dict:
        """导出报告（JSON + Markdown）"""
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = os.path.join(output_dir, f"teacher_report_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        md_path = os.path.join(output_dir, f"teacher_report_{ts}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())

        return {"json": json_path, "markdown": md_path}

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        for start_ch, end_ch in [('[', ']'), ('{', '}')]:
            count = 0
            start = -1
            for i, ch in enumerate(text):
                if ch == start_ch:
                    if start == -1:
                        start = i
                    count += 1
                elif ch == end_ch:
                    count -= 1
                    if count == 0 and start != -1:
                        return text[start:i + 1]
        return text

    def _fallback_question_set(self, project_text: str) -> list:
        """LLM JSON解析失败时，用QuestionGeneratorSkill兜底"""
        result = self.qgen.generate(project_text)
        items = []
        for q in result.basic_questions:
            items.append(QuestionSetItem(question=q.question, reference_answer="请参考项目材料", category="基础", difficulty="中"))
        for q in result.technical_questions:
            items.append(QuestionSetItem(question=q.question, reference_answer="请参考项目材料", category="技术", difficulty="高"))
        for q in result.improvement_questions:
            items.append(QuestionSetItem(question=q.question, reference_answer="请参考项目材料", category="改进", difficulty="中"))
        return items