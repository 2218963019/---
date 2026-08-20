"""
Skill3: 方案优化Skill
根据项目材料，输出系统改进思路、算法优化方向、Demo升级建议，
贴合AI知识图谱、多模态视觉方向。
"""

import re
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class OptimizationItem:
    """单条优化建议"""
    direction: str
    suggestion: str
    relevance: str = "高"


@dataclass
class OptimizationResult:
    """方案优化结果"""
    system_improvements: list = field(default_factory=list)
    algorithm_optimizations: list = field(default_factory=list)
    demo_upgrades: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "系统改进思路": [{"方向": s.direction, "建议": s.suggestion, "相关度": s.relevance} for s in self.system_improvements],
            "算法优化方向": [{"方向": s.direction, "建议": s.suggestion, "相关度": s.relevance} for s in self.algorithm_optimizations],
            "Demo升级建议": [{"方向": s.direction, "建议": s.suggestion, "相关度": s.relevance} for s in self.demo_upgrades],
        }

    def summary(self) -> str:
        lines = ["===== 方案优化建议 ====="]
        lines.append(f"\n【系统改进思路】({len(self.system_improvements)}条)")
        for i, s in enumerate(self.system_improvements, 1):
            lines.append(f"  {i}. [{s.direction}] {s.suggestion} (相关度:{s.relevance})")
        lines.append(f"\n【算法优化方向】({len(self.algorithm_optimizations)}条)")
        for i, s in enumerate(self.algorithm_optimizations, 1):
            lines.append(f"  {i}. [{s.direction}] {s.suggestion} (相关度:{s.relevance})")
        lines.append(f"\n【Demo升级建议】({len(self.demo_upgrades)}条)")
        for i, s in enumerate(self.demo_upgrades, 1):
            lines.append(f"  {i}. [{s.direction}] {s.suggestion} (相关度:{s.relevance})")
        return "\n".join(lines)


class SolutionOptimizerSkill:
    """方案优化Skill，贴合AI知识图谱与多模态视觉方向"""

    SYSTEM_RULES = [
        (r"知识图谱|图谱|KG", [
            OptimizationItem("知识图谱", "引入增量式知识图谱更新机制，支持动态实体/关系插入", "高"),
            OptimizationItem("知识图谱", "采用图神经网络(GNN)对图谱结构进行深度推理增强", "高"),
            OptimizationItem("知识图谱", "增加跨模态对齐模块，将视觉特征映射到图谱语义空间", "高"),
        ]),
        (r"多模态|视觉|图像|视频|VLM", [
            OptimizationItem("多模态视觉", "集成视觉语言模型(VLM)实现图文联合理解", "高"),
            OptimizationItem("多模态视觉", "引入视觉注意力机制，提升关键区域特征提取能力", "高"),
            OptimizationItem("多模态视觉", "增加多尺度视觉特征融合策略（FPN/PAN）", "中"),
        ]),
        (r"系统|架构|服务|部署", [
            OptimizationItem("系统架构", "采用微服务架构拆分，提升系统可扩展性", "高"),
            OptimizationItem("系统架构", "引入异步消息队列解耦各模块，提升吞吐量", "中"),
            OptimizationItem("系统架构", "增加缓存层（Redis）加速高频查询响应", "中"),
        ]),
        (r"检索|搜索|查询|RAG", [
            OptimizationItem("检索增强", "引入RAG架构，结合向量检索与大模型生成", "高"),
            OptimizationItem("检索增强", "采用混合检索策略（稀疏+稠密向量）提升召回率", "高"),
            OptimizationItem("检索增强", "增加查询改写与扩展模块，提升检索语义匹配度", "中"),
        ]),
    ]

    ALGORITHM_RULES = [
        (r"知识图谱|实体|关系|推理", [
            OptimizationItem("图谱推理", "采用TransE/TransH等知识表示学习增强推理能力", "高"),
            OptimizationItem("图谱推理", "引入规则推理与图神经网络混合推理框架", "高"),
            OptimizationItem("图谱推理", "增加可解释性推理路径输出，提升结果可信度", "中"),
        ]),
        (r"多模态|视觉|图像|融合", [
            OptimizationItem("多模态融合", "采用Cross-Attention实现图文深度交互融合", "高"),
            OptimizationItem("多模态融合", "引入CLIP模型进行视觉-语言对齐预训练", "高"),
            OptimizationItem("多模态融合", "增加模态缺失鲁棒处理，支持单模态降级推理", "中"),
        ]),
        (r"算法|模型|训练|深度学习", [
            OptimizationItem("算法优化", "引入对比学习增强表征质量", "高"),
            OptimizationItem("算法优化", "采用LoRA/QLoRA进行参数高效微调", "高"),
            OptimizationItem("算法优化", "增加模型蒸馏方案，降低推理延迟", "中"),
        ]),
        (r"NLP|文本|语义|理解", [
            OptimizationItem("语义理解", "引入大语言模型(LLM)增强语义理解深度", "高"),
            OptimizationItem("语义理解", "增加长文本处理能力（滑动窗口/层次化注意力）", "中"),
            OptimizationItem("语义理解", "采用指令微调提升任务适配能力", "中"),
        ]),
    ]

    DEMO_RULES = [
        (r"知识图谱|图谱|可视化", [
            OptimizationItem("图谱可视化", "增加交互式知识图谱可视化面板（D3.js/ECharts）", "高"),
            OptimizationItem("图谱可视化", "支持图谱动态演化动画展示", "中"),
        ]),
        (r"多模态|视觉|图像|视频", [
            OptimizationItem("视觉交互", "增加图像/视频上传与实时分析演示功能", "高"),
            OptimizationItem("视觉交互", "引入可视化注意力热力图展示模型关注区域", "高"),
            OptimizationItem("视觉交互", "增加图文对比展示面板，直观呈现多模态对齐效果", "中"),
        ]),
        (r"问答|对话|chat|Chat", [
            OptimizationItem("交互体验", "增加流式输出与打字机效果，提升对话体验", "高"),
            OptimizationItem("交互体验", "支持多轮对话上下文展示与历史回溯", "中"),
        ]),
        (r"API|接口|前端|web", [
            OptimizationItem("前端升级", "采用React/Vue3重构前端，提升交互流畅度", "高"),
            OptimizationItem("前端升级", "增加API文档自动生成（Swagger/FastAPI）", "中"),
            OptimizationItem("前端升级", "引入WebSocket实现实时推送与进度反馈", "中"),
        ]),
    ]

    def __init__(self):
        self._result: Optional[OptimizationResult] = None

    def optimize(self, project_text: str) -> OptimizationResult:
        """
        根据项目材料生成优化建议。

        Args:
            project_text: 项目全文文本

        Returns:
            OptimizationResult 包含三类优化建议
        """
        result = OptimizationResult()
        result.system_improvements = self._match_rules(project_text, self.SYSTEM_RULES)
        result.algorithm_optimizations = self._match_rules(project_text, self.ALGORITHM_RULES)
        result.demo_upgrades = self._match_rules(project_text, self.DEMO_RULES)

        if not result.system_improvements:
            result.system_improvements = self._default_system()
        if not result.algorithm_optimizations:
            result.algorithm_optimizations = self._default_algorithm()
        if not result.demo_upgrades:
            result.demo_upgrades = self._default_demo()

        self._result = result
        return result

    def _match_rules(self, text: str, rules: list) -> list:
        """根据规则匹配生成优化建议"""
        items = []
        seen = set()
        for pattern, suggestions in rules:
            if re.search(pattern, text, re.IGNORECASE):
                for s in suggestions:
                    key = (s.direction, s.suggestion)
                    if key not in seen:
                        items.append(s)
                        seen.add(key)
        return items

    @staticmethod
    def _default_system() -> list:
        return [
            OptimizationItem("通用", "引入模块化设计，支持功能热插拔", "高"),
            OptimizationItem("通用", "增加日志监控与性能分析模块", "中"),
            OptimizationItem("通用", "完善异常处理与降级策略", "中"),
        ]

    @staticmethod
    def _default_algorithm() -> list:
        return [
            OptimizationItem("通用", "评估引入预训练大模型替代现有小模型的可行性", "高"),
            OptimizationItem("通用", "增加消融实验验证各模块贡献", "中"),
            OptimizationItem("通用", "引入自动化超参搜索（Optuna/Ray）", "中"),
        ]

    @staticmethod
    def _default_demo() -> list:
        return [
            OptimizationItem("通用", "增加交互式Demo展示页面", "高"),
            OptimizationItem("通用", "支持一键部署脚本（Docker Compose）", "中"),
            OptimizationItem("通用", "增加性能基准测试与对比展示", "中"),
        ]

    def get_result(self) -> Optional[OptimizationResult]:
        """获取最近一次优化结果"""
        return self._result

    def export_suggestions(self, file_path: str) -> None:
        """
        将优化建议导出到文本文件。

        Args:
            file_path: 输出文件路径
        """
        if not self._result:
            raise RuntimeError("尚未生成优化建议，请先调用optimize()")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self._result.summary())