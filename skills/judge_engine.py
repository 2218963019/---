"""
评委人设引擎 + 答辩Session管理
核心差异化模块：多风格评委模拟、动态追问、评分、答辩全流程管理。
"""

import re
import json
import os
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from skills.llm_client import LLMClient, ConversationHistory
from skills.rag_engine import RAGEngine


# ============================================================
# 1. 评委人设定义
# ============================================================

@dataclass
class JudgePersona:
    """评委人设"""
    name: str
    style: str
    system_prompt: str
    question_focus: str
    followup_strategy: str
    tone: str
    scoring_weights: dict = field(default_factory=dict)


JUDGE_PRESETS = {
    "strict_tech": JudgePersona(
        name="严厉技术型评委",
        style="strict_tech",
        system_prompt=(
            "你是一位严谨的技术评审专家，专注于考察项目的技术深度和可行性。\n"
            "你的提问风格：\n"
            "- 直击技术核心，追问算法细节、实验数据、对比基线\n"
            "- 对模糊回答零容忍，要求精确数据和具体方案\n"
            "- 会指出技术方案中的潜在缺陷和风险\n"
            "- 语气严肃专业，不轻易给出肯定\n"
            "- 每次只问一个问题，问题要深入具体"
        ),
        question_focus="算法原理、实验设计、技术对比、数据支撑、实现细节",
        followup_strategy="深挖技术细节，要求量化证据，质疑方案合理性",
        tone="严肃、精确、质疑",
        scoring_weights={"技术深度": 0.35, "创新性": 0.25, "可行性": 0.20, "表达清晰度": 0.10, "实验充分性": 0.10},
    ),
    "practical": JudgePersona(
        name="务实应用型评委",
        style="practical",
        system_prompt=(
            "你是一位注重落地的应用型评审专家，关注项目的实际价值和可部署性。\n"
            "你的提问风格：\n"
            "- 追问应用场景、目标用户、市场需求\n"
            "- 关注成本效益、部署难度、运维方案\n"
            "- 会对比竞品和现有解决方案\n"
            "- 语气务实直接，喜欢用'具体来说''举个例子'\n"
            "- 每次只问一个问题，聚焦落地可行性"
        ),
        question_focus="应用场景、用户需求、成本效益、竞品对比、部署方案",
        followup_strategy="追问落地细节，要求具体场景和数据，质疑商业可行性",
        tone="务实、直接、结果导向",
        scoring_weights={"应用价值": 0.30, "可行性": 0.25, "创新性": 0.20, "技术深度": 0.15, "表达清晰度": 0.10},
    ),
    "strategic": JudgePersona(
        name="战略宏观型评委",
        style="strategic",
        system_prompt=(
            "你是一位站在行业高度的评审专家，关注项目的战略价值和长期潜力。\n"
            "你的提问风格：\n"
            "- 从行业趋势和技术发展角度提问\n"
            "- 追问项目的护城河、壁垒和可持续性\n"
            "- 关注团队能力和资源匹配\n"
            "- 会将项目放在更大的技术版图中定位\n"
            "- 语气宏观深远，喜欢用'从长远来看''行业趋势'"
        ),
        question_focus="行业趋势、技术壁垒、团队实力、长期规划、差异化优势",
        followup_strategy="拔高视角，追问战略意义和长期价值，考察格局和视野",
        tone="宏观、深远、格局导向",
        scoring_weights={"创新性": 0.30, "战略价值": 0.25, "可行性": 0.20, "团队实力": 0.15, "表达清晰度": 0.10},
    ),
    "encouraging": JudgePersona(
        name="鼓励引导型评委",
        style="encouraging",
        system_prompt=(
            "你是一位温和的评审专家，倾向于先肯定再建议，帮助学生展现最好的一面。\n"
            "你的提问风格：\n"
            "- 先肯定项目亮点，再温和地指出改进方向\n"
            "- 用引导式提问帮助学生深入思考\n"
            "- 避免直接否定，用'有没有考虑过''可以进一步探讨'等措辞\n"
            "- 给学生充分表达的空间\n"
            "- 语气温暖支持，但仍有专业深度"
        ),
        question_focus="项目亮点、改进空间、未来方向、团队协作、个人成长",
        followup_strategy="先肯定再引导，用开放式问题拓展思考，温和指出不足",
        tone="温和、支持、引导式",
        scoring_weights={"创新性": 0.25, "表达清晰度": 0.25, "可行性": 0.20, "技术深度": 0.15, "应用价值": 0.15},
    ),
    "mixed": JudgePersona(
        name="混合随机型评委组",
        style="mixed",
        system_prompt=(
            "你是一个由3位不同风格评委组成的答辩小组：\n"
            "- 技术专家：关注技术深度和实验严谨性\n"
            "- 应用专家：关注落地价值和用户体验\n"
            "- 战略专家：关注行业趋势和长期潜力\n"
            "每次提问时，随机选择一位评委的视角，模拟真实答辩中评委轮番提问的场景。\n"
            "在提问前标注当前发言的评委角色，如[技术专家]、[应用专家]。"
        ),
        question_focus="全方位考察，技术+应用+战略轮番提问",
        followup_strategy="不同评委从各自角度追问，形成多维度压力测试",
        tone="多样化、轮番切换",
        scoring_weights={"技术深度": 0.25, "创新性": 0.20, "应用价值": 0.20, "可行性": 0.20, "表达清晰度": 0.15},
    ),
}


# ============================================================
# 2. 评审标准与评分
# ============================================================

EVALUATION_CRITERIA = {
    "大创立项": {
        "创新性": "项目是否有新颖的研究思路或技术方案，与现有工作相比的差异化",
        "可行性": "技术路线是否合理，是否有足够的资源和能力完成",
        "技术深度": "核心技术方案是否有深度，是否理解原理而非简单调用",
        "应用价值": "研究成果是否有实际应用前景和社会价值",
        "表达清晰度": "答辩表达是否清晰、逻辑是否严谨、回答是否切题",
    },
    "竞赛路演": {
        "技术深度": "核心技术方案的深度和完整性",
        "创新性": "与竞品和现有方案的差异化优势",
        "应用价值": "产品/方案的市场需求和商业潜力",
        "可行性": "技术实现和商业落地的可行性",
        "表达清晰度": "路演表现、Demo演示效果、回答质量",
    },
}


@dataclass
class DimensionScore:
    """单维度评分"""
    dimension: str
    score: float
    comment: str


@dataclass
class EvaluationResult:
    """评审结果"""
    scores: list = field(default_factory=list)
    total_score: float = 0.0
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "各维度评分": [{"维度": s.dimension, "分数": s.score, "评语": s.comment} for s in self.scores],
            "总分": self.total_score,
            "优势": self.strengths,
            "不足": self.weaknesses,
            "改进建议": self.suggestions,
        }

    def summary(self) -> str:
        lines = ["===== 评审结果 ====="]
        lines.append(f"\n总分: {self.total_score:.1f}/100")
        lines.append("\n各维度评分:")
        for s in self.scores:
            bar = "█" * int(s.score / 5) + "░" * (20 - int(s.score / 5))
            lines.append(f"  {s.dimension}: {bar} {s.score:.1f}/100 — {s.comment}")
        if self.strengths:
            lines.append(f"\n优势: {'; '.join(self.strengths)}")
        if self.weaknesses:
            lines.append(f"\n不足: {'; '.join(self.weaknesses)}")
        if self.suggestions:
            lines.append(f"\n改进建议:")
            for i, s in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {s}")
        return "\n".join(lines)


# ============================================================
# 3. 答辩Session管理
# ============================================================

@dataclass
class DefenseTurn:
    """单轮答辩记录"""
    turn_id: int
    question: str
    answer: str = ""
    followup_questions: list = field(default_factory=list)
    followup_answers: list = field(default_factory=list)
    judge_comment: str = ""


@dataclass
class DefenseSession:
    """答辩会话"""
    session_id: str = ""
    persona: Optional[JudgePersona] = None
    scenario: str = "大创立项"
    turns: list = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None
    project_summary: str = ""
    created_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "persona": self.persona.name if self.persona else "",
            "scenario": self.scenario,
            "turns": [
                {
                    "轮次": t.turn_id,
                    "问题": t.question,
                    "回答": t.answer,
                    "追问": t.followup_questions,
                    "追问回答": t.followup_answers,
                    "评委点评": t.judge_comment,
                }
                for t in self.turns
            ],
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
        }


# ============================================================
# 4. 评委引擎主类
# ============================================================

class JudgeEngine:
    """
    评委引擎：整合LLM + RAG + 人设，驱动完整答辩流程。

    使用方式:
        engine = JudgeEngine(llm_client=client, rag_engine=rag)
        session = engine.start_session(persona="strict_tech", scenario="大创立项")
        question = engine.ask_first_question(session)
        engine.answer(session, "我们的创新点是...")
        followup = engine.followup(session)
        result = engine.evaluate(session)
    """

    FIRST_QUESTION_PROMPT = (
        "基于以下项目材料，提出你的第一个答辩问题。\n"
        "要求：\n"
        "- 问题要体现你的评审风格和关注重点\n"
        "- 只问一个问题，要深入具体\n"
        "- 不要问'请介绍一下项目'这类泛泛的问题\n\n"
        "项目材料：\n{context}\n\n"
        "请直接提出你的问题："
    )

    FOLLOWUP_PROMPT = (
        "你刚才问了：{question}\n"
        "学生回答：{answer}\n\n"
        "请分析学生的回答，决定是否需要追问。\n"
        "追问策略：\n"
        "- 如果回答模糊或避重就轻，追问具体细节\n"
        "- 如果回答与材料矛盾，指出矛盾并追问\n"
        "- 如果回答有深度，可以切换到下一个关注点\n"
        "- 最多追问1个问题\n\n"
        "项目材料供参考：\n{context}\n\n"
        "请以JSON格式输出：\n"
        '{{"need_followup": true/false, "followup_question": "追问内容", "comment": "对回答的简短点评"}}'
    )

    EVALUATE_PROMPT = (
        "请对本次答辩进行全面评审。\n\n"
        "评审标准：\n{criteria}\n\n"
        "答辩记录：\n{transcript}\n\n"
        "项目材料：\n{context}\n\n"
        "请严格按以下JSON格式输出评审结果（不要输出任何其他内容，不要用markdown代码块包裹）：\n"
        '{{"scores": {{"技术深度": 85, "创新性": 78, "可行性": 72, "应用价值": 75, "表达清晰度": 80}}, "strengths": ["优势1", "优势2"], "weaknesses": ["不足1", "不足2"], "suggestions": ["建议1", "建议2"]}}\n'
        "评分规则：每个维度0-100分。scores中的维度名必须与评审标准中的维度名完全一致。"
    )

    def __init__(
        self,
        llm_client: LLMClient,
        rag_engine: Optional[RAGEngine] = None,
        max_followups: int = 2,
        max_turns: int = 8,
    ):
        self.llm = llm_client
        self.rag = rag_engine
        self.max_followups = max_followups
        self.max_turns = max_turns
        self._sessions: dict = {}

    def start_session(
        self,
        persona: str = "strict_tech",
        scenario: str = "大创立项",
        project_text: str = "",
    ) -> DefenseSession:
        """
        开始一个答辩会话。

        Args:
            persona: 评委人设key (strict_tech/practical/strategic/encouraging/mixed)
            scenario: 答辩场景 (大创立项/竞赛路演)
            project_text: 项目全文（可选，如有RAG引擎则自动索引）

        Returns:
            DefenseSession
        """
        if persona not in JUDGE_PRESETS:
            raise ValueError(f"未知评委人设: {persona}，可选: {list(JUDGE_PRESETS.keys())}")

        session = DefenseSession(
            session_id=f"defense_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            persona=JUDGE_PRESETS[persona],
            scenario=scenario,
            project_summary=project_text[:200] if project_text else "",
            created_at=datetime.now().isoformat(),
        )

        if project_text and self.rag:
            self.rag.index_text(project_text, source="session_project", strategy="section")

        self._sessions[session.session_id] = session
        return session

    def _get_context(self, query: str, top_k: int = 5) -> str:
        """获取RAG检索上下文"""
        if self.rag and self.rag.store.size > 0:
            return self.rag.retrieve_with_context(query, top_k=top_k)
        return ""

    def ask_first_question(self, session: DefenseSession) -> str:
        """
        评委提出第一个问题。

        Args:
            session: 答辩会话

        Returns:
            第一个问题
        """
        context = self._get_context("项目概述 创新点 技术路线", top_k=5)
        if not context and session.project_summary:
            context = session.project_summary

        prompt = self.FIRST_QUESTION_PROMPT.format(context=context or "暂无项目材料")

        messages = [
            {"role": "system", "content": session.persona.system_prompt},
            {"role": "user", "content": prompt},
        ]

        resp = self.llm.chat(messages)
        question = resp.content.strip()

        turn = DefenseTurn(turn_id=1, question=question)
        session.turns.append(turn)
        return question

    def answer(self, session: DefenseSession, answer: str) -> None:
        """
        学生回答当前问题。

        Args:
            session: 答辩会话
            answer: 学生回答
        """
        if not session.turns:
            raise RuntimeError("评委尚未提问，请先调用ask_first_question()")

        current_turn = session.turns[-1]
        if current_turn.answer and not current_turn.followup_questions:
            raise RuntimeError("当前轮次已回答完毕，请调用next_question()进入下一轮")

        if not current_turn.answer:
            current_turn.answer = answer
        else:
            current_turn.followup_answers.append(answer)

    def followup(self, session: DefenseSession) -> Optional[str]:
        """
        评委根据学生回答决定是否追问。

        Args:
            session: 答辩会话

        Returns:
            追问问题（None表示不追问）
        """
        current_turn = session.turns[-1]
        if not current_turn.answer:
            raise RuntimeError("学生尚未回答，请先调用answer()")

        if len(current_turn.followup_questions) >= self.max_followups:
            return None

        context = self._get_context(current_turn.question, top_k=3)

        prompt = self.FOLLOWUP_PROMPT.format(
            question=current_turn.question,
            answer=current_turn.answer,
            context=context or "暂无材料",
        )

        messages = [
            {"role": "system", "content": session.persona.system_prompt},
            {"role": "user", "content": prompt},
        ]

        resp = self.llm.chat(messages, temperature=0.3)

        json_str = self._extract_json(resp.content.strip())
        try:
            result = json.loads(json_str)
            if result.get("need_followup") and result.get("followup_question"):
                question = result["followup_question"]
                current_turn.followup_questions.append(question)
                current_turn.judge_comment = result.get("comment", "")
                return question
            else:
                current_turn.judge_comment = result.get("comment", "")
                return None
        except json.JSONDecodeError:
            if "?" in resp.content or "？" in resp.content:
                current_turn.followup_questions.append(resp.content.strip())
                return resp.content.strip()
            current_turn.judge_comment = resp.content.strip()
            return None

    def next_question(self, session: DefenseSession) -> Optional[str]:
        """
        评委提出下一个新问题（非追问）。

        Args:
            session: 答辩会话

        Returns:
            新问题（None表示答辩结束）
        """
        if len(session.turns) >= self.max_turns:
            return None

        asked = []
        for t in session.turns:
            asked.append(f"Q: {t.question}")
            if t.answer:
                asked.append(f"A: {t.answer}")

        context = self._get_context("答辩考察重点", top_k=5)

        prompt = (
            f"以下是已有的答辩记录：\n{chr(10).join(asked)}\n\n"
            f"项目材料：\n{context or session.project_summary or '暂无材料'}\n\n"
            f"请提出一个全新的问题，不要重复已问过的内容。\n"
            f"关注重点：{session.persona.question_focus}\n"
            f"只问一个问题，直接输出问题："
        )

        messages = [
            {"role": "system", "content": session.persona.system_prompt},
            {"role": "user", "content": prompt},
        ]

        resp = self.llm.chat(messages)
        question = resp.content.strip()

        turn = DefenseTurn(turn_id=len(session.turns) + 1, question=question)
        session.turns.append(turn)
        return question

    def evaluate(self, session: DefenseSession) -> EvaluationResult:
        """
        答辩结束后进行评审打分。

        Args:
            session: 答辩会话

        Returns:
            EvaluationResult
        """
        criteria = EVALUATION_CRITERIA.get(session.scenario, EVALUATION_CRITERIA["大创立项"])
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in criteria.items())

        transcript_lines = []
        for t in session.turns:
            transcript_lines.append(f"【第{t.turn_id}轮】")
            transcript_lines.append(f"评委: {t.question}")
            if t.answer:
                transcript_lines.append(f"学生: {t.answer}")
            for i, fq in enumerate(t.followup_questions):
                transcript_lines.append(f"评委追问: {fq}")
                if i < len(t.followup_answers):
                    transcript_lines.append(f"学生: {t.followup_answers[i]}")
            if t.judge_comment:
                transcript_lines.append(f"评委点评: {t.judge_comment}")
        transcript = "\n".join(transcript_lines)

        context = self._get_context("项目全貌", top_k=5)

        prompt = self.EVALUATE_PROMPT.format(
            criteria=criteria_text,
            transcript=transcript,
            context=context or session.project_summary or "暂无材料",
        )

        messages = [
            {"role": "system", "content": session.persona.system_prompt},
            {"role": "user", "content": prompt},
        ]

        resp = self.llm.chat(messages, temperature=0.2)

        raw = resp.content.strip()
        json_str = self._extract_json(raw)

        try:
            data = json.loads(json_str)
            result = EvaluationResult()

            weights = session.persona.scoring_weights
            weighted_total = 0.0
            weight_sum = 0.0

            for dim, score in data.get("scores", {}).items():
                s = float(score)
                result.scores.append(DimensionScore(dimension=dim, score=s, comment=""))
                w = weights.get(dim, 0.1)
                weighted_total += s * w
                weight_sum += w

            result.total_score = weighted_total / weight_sum if weight_sum > 0 else 0
            result.strengths = data.get("strengths", [])
            result.weaknesses = data.get("weaknesses", [])
            result.suggestions = data.get("suggestions", [])

        except (json.JSONDecodeError, ValueError):
            result = self._fallback_evaluate(session, raw)

        session.evaluation = result
        session.finished_at = datetime.now().isoformat()
        return result

    def get_session(self, session_id: str) -> Optional[DefenseSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list:
        return list(self._sessions.keys())

    def export_session(self, session: DefenseSession, file_path: str) -> None:
        """导出答辩记录到JSON"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def list_personas() -> dict:
        """列出所有可用评委人设"""
        return {k: p.name for k, p in JUDGE_PRESETS.items()}

    @staticmethod
    def list_scenarios() -> list:
        """列出所有可用答辩场景"""
        return list(EVALUATION_CRITERIA.keys())

    @staticmethod
    def _extract_json(text: str) -> str:
        """从LLM输出中提取JSON（兼容markdown代码块包裹）"""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

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

    def _fallback_evaluate(self, session: DefenseSession, raw_text: str) -> EvaluationResult:
        """JSON解析失败时的兜底评审：用LLM逐维度打分"""
        criteria = EVALUATION_CRITERIA.get(session.scenario, EVALUATION_CRITERIA["大创立项"])
        result = EvaluationResult()
        weights = session.persona.scoring_weights
        weighted_total = 0.0
        weight_sum = 0.0

        for dim, desc in criteria.items():
            prompt = (
                f"评审维度：{dim}（{desc}）\n"
                f"答辩记录摘要：{raw_text[:500]}\n"
                f"请只输出一个0-100的整数分数，不要输出其他内容："
            )
            try:
                resp = self.llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=10,
                )
                score = float(re.search(r'\d+', resp.content).group())
                score = max(0, min(100, score))
            except Exception:
                score = 60.0

            result.scores.append(DimensionScore(dimension=dim, score=score, comment=""))
            w = weights.get(dim, 0.1)
            weighted_total += score * w
            weight_sum += w

        result.total_score = weighted_total / weight_sum if weight_sum > 0 else 0
        result.suggestions = ["评审完成（兜底模式）"]
        return result