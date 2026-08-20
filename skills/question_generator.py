"""
Skill2: 答辩问题生成Skill
输入项目文本内容，自动提炼评委高频提问，
区分基础项目问题、技术细节问题、未来改进方向问题。
"""

import re
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class QuestionItem:
    """单个答辩问题"""
    category: str
    question: str
    priority: int = 0


@dataclass
class QuestionResult:
    """答辩问题生成结果"""
    basic_questions: list = field(default_factory=list)
    technical_questions: list = field(default_factory=list)
    improvement_questions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "基础项目问题": [q.question for q in self.basic_questions],
            "技术细节问题": [q.question for q in self.technical_questions],
            "未来改进方向问题": [q.question for q in self.improvement_questions],
        }

    def summary(self) -> str:
        lines = ["===== 答辩问题生成结果 ====="]
        lines.append(f"\n【基础项目问题】({len(self.basic_questions)}个)")
        for i, q in enumerate(self.basic_questions, 1):
            lines.append(f"  {i}. {q.question}")
        lines.append(f"\n【技术细节问题】({len(self.technical_questions)}个)")
        for i, q in enumerate(self.technical_questions, 1):
            lines.append(f"  {i}. {q.question}")
        lines.append(f"\n【未来改进方向问题】({len(self.improvement_questions)}个)")
        for i, q in enumerate(self.improvement_questions, 1):
            lines.append(f"  {i}. {q.question}")
        return "\n".join(lines)


class QuestionGeneratorSkill:
    """答辩问题生成Skill"""

    BASIC_PATTERNS = [
        (r"项目背景|研究背景|立项依据", "请简述项目的立项背景和核心动机是什么？"),
        (r"研究目标|项目目标", "项目的主要研究目标是什么？是否已全部达成？"),
        (r"创新点|创新性|特色", "项目的核心创新点是什么？与现有方案相比有何优势？"),
        (r"应用场景|应用价值|实际应用", "项目的实际应用场景有哪些？落地可行性如何？"),
        (r"团队|成员|分工", "请介绍团队成员分工及各自贡献？"),
        (r"经费|预算|资金", "项目经费使用情况如何？是否合理？"),
    ]

    TECHNICAL_PATTERNS = [
        (r"算法|模型|方法", "项目中使用的核心算法/模型是什么？为何选择该方案？"),
        (r"数据集|数据来源|数据预处理", "训练/测试使用的数据集是什么？数据预处理流程是怎样的？"),
        (r"实验|评估|指标|准确率", "实验评估采用了哪些指标？结果与基线对比如何？"),
        (r"架构|系统|框架|技术栈", "系统的整体技术架构是怎样的？各模块如何协作？"),
        (r"知识图谱|图谱|实体|关系", "知识图谱的构建方法是什么？实体和关系如何定义？"),
        (r"多模态|视觉|图像|视频", "多模态信息的融合方式是什么？视觉特征如何提取？"),
        (r"接口|API|部署|服务", "系统的部署方案是什么？API接口如何设计？"),
    ]

    IMPROVEMENT_PATTERNS = [
        (r"局限|不足|缺陷|挑战", "当前方案的主要局限性是什么？如何改进？"),
        (r"扩展|拓展|推广|泛化", "项目成果如何推广到更广泛的应用场景？"),
        (r"优化|提升|改进|增强", "算法/系统层面还有哪些优化空间？"),
        (r"未来|下一步|后续|规划", "未来的研究/开发计划是什么？"),
        (r"鲁棒|稳定性|可靠性", "系统的鲁棒性和稳定性如何保障？有何改进计划？"),
        (r"实时|性能|效率|速度", "系统的实时性能如何？有哪些加速优化方向？"),
    ]

    def __init__(self):
        self._result: Optional[QuestionResult] = None

    def generate(self, project_text: str, custom_keywords: Optional[dict] = None) -> QuestionResult:
        """
        根据项目文本生成答辩问题。

        Args:
            project_text: 项目全文文本
            custom_keywords: 自定义关键词映射，格式为
                {"基础": [(pattern, question), ...], "技术": [...], "改进": [...]}

        Returns:
            QuestionResult 包含三类问题
        """
        basic_patterns = custom_keywords.get("基础", self.BASIC_PATTERNS) if custom_keywords else self.BASIC_PATTERNS
        tech_patterns = custom_keywords.get("技术", self.TECHNICAL_PATTERNS) if custom_keywords else self.TECHNICAL_PATTERNS
        imp_patterns = custom_keywords.get("改进", self.IMPROVEMENT_PATTERNS) if custom_keywords else self.IMPROVEMENT_PATTERNS

        result = QuestionResult()
        result.basic_questions = self._match_questions(project_text, basic_patterns, "基础")
        result.technical_questions = self._match_questions(project_text, tech_patterns, "技术")
        result.improvement_questions = self._match_questions(project_text, imp_patterns, "改进")

        self._result = result
        return result

    def _match_questions(self, text: str, patterns: list, category: str) -> list:
        """根据模式匹配生成问题"""
        questions = []
        seen = set()
        for pattern, question in patterns:
            if re.search(pattern, text):
                if question not in seen:
                    questions.append(QuestionItem(category=category, question=question, priority=len(questions)))
                    seen.add(question)

        fallback = {
            "基础": [
                "请用3分钟介绍你的项目？",
                "项目解决了什么痛点问题？",
                "项目与同类研究相比有何差异化优势？",
            ],
            "技术": [
                "核心技术方案的设计思路是什么？",
                "关键技术难点是如何攻克的？",
                "实验结果是否具有可复现性？",
            ],
            "改进": [
                "如果重新做这个项目，你会做哪些调整？",
                "项目还有哪些未解决的技术挑战？",
                "如何将成果转化为实际产品？",
            ],
        }

        if not questions:
            for q in fallback.get(category, []):
                questions.append(QuestionItem(category=category, question=q))

        return questions

    def get_result(self) -> Optional[QuestionResult]:
        """获取最近一次生成结果"""
        return self._result

    def export_questions(self, file_path: str) -> None:
        """
        将问题导出到文本文件。

        Args:
            file_path: 输出文件路径
        """
        if not self._result:
            raise RuntimeError("尚未生成问题，请先调用generate()")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self._result.summary())