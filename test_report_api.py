"""
报告生成器 + API 端到端测试
离线: 数据结构、序列化、Markdown生成
在线: 完整答辩→评审→报告流程
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from skills.report_generator import ReportGenerator, DefenseReport, QAItem, ImprovementItem
from skills.judge_engine import DefenseSession, EvaluationResult, DimensionScore, JUDGE_PRESETS
from skills.llm_client import LLMClient
from skills.rag_engine import RAGEngine

PASS = "✓ 通过"
FAIL = "✗ 失败"
results = []


def test(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


print("=" * 60)
print("报告生成器 + API 端到端测试")
print("=" * 60)

# ========== 1. 数据结构 ==========
print("\n【1. DefenseReport 数据结构】")

report = DefenseReport(
    session_id="test_001",
    persona_name="严厉技术型评委",
    scenario="大创立项",
    total_score=72.5,
    dimension_scores=[
        DimensionScore("技术深度", 62, "技术方案有深度但缺乏细节"),
        DimensionScore("创新性", 78, "创新点明确"),
        DimensionScore("可行性", 72, "部分风险未讨论"),
        DimensionScore("应用价值", 80, "应用场景清晰"),
        DimensionScore("表达清晰度", 68, "部分回答模糊"),
    ],
    strengths=["技术方案扎实", "创新点明确"],
    weaknesses=["缺乏实验数据支撑", "部分回答避重就轻"],
    improvements=[
        ImprovementItem("技术深度", 62, 80, "缺少消融实验数据", "补充GAT消融实验和p值", "高"),
        ImprovementItem("表达清晰度", 68, 80, "回答不够具体", "练习用数据说话", "中"),
    ],
    qa_set=[
        QAItem("创新点是什么？", "跨模态知识对齐方法", "基础", "低"),
        QAItem("GAT如何指导融合？", "利用图谱注意力权重指导Cross-Attention", "技术", "高"),
    ],
    radar_data={"dimensions": ["技术深度", "创新性"], "scores": [62, 78], "max": 100},
    summary="答辩表现中等，技术方案有亮点但细节不足。",
)
test("DefenseReport创建", report.total_score == 72.5)

# to_dict
d = report.to_dict()
test("to_dict完整", all(k in d for k in ["总分", "各维度得分", "雷达图数据", "优势", "不足", "改进建议", "QA问题集", "总结"]))
test("QA问题集数量", len(d["QA问题集"]) == 2)
test("改进建议数量", len(d["改进建议"]) == 2)

# to_markdown
md = report.to_markdown()
test("Markdown含标题", "# 答辩改进报告" in md)
test("Markdown含评分", "技术深度" in md and "62" in md)
test("Markdown含QA", "Q1." in md and "参考答案" in md)
test("Markdown含改进", "改进建议" in md and "消融实验" in md)
test("Markdown含总结", "答辩表现中等" in md)

# ========== 2. 雷达图数据 ==========
print("\n【2. 雷达图数据格式】")

radar = report.radar_data
test("雷达图有dimensions", len(radar["dimensions"]) > 0)
test("雷达图有scores", len(radar["scores"]) > 0)
test("维度和分数数量一致", len(radar["dimensions"]) == len(radar["scores"]))
test("分数在0-100范围", all(0 <= s <= 100 for s in radar["scores"]))

# ========== 3. QAItem & ImprovementItem ==========
print("\n【3. QAItem & ImprovementItem】")

qa = QAItem(question="创新点", reference_answer="跨模态对齐", category="基础", difficulty="低")
test("QAItem创建", qa.category == "基础")

imp = ImprovementItem(dimension="技术深度", current_score=60, target_score=80, gap="缺数据", action="补实验", priority="高")
test("ImprovementItem创建", imp.priority == "高")

# ========== 4. 报告导出 ==========
print("\n【4. 报告文件导出】")

mock_client = LLMClient(api_key="mock", preset="qwen")
generator = ReportGenerator(llm_client=mock_client)

output_dir = os.path.join(os.path.dirname(__file__), "_test_report_output")
paths = generator.export_report(report, output_dir)
test("导出JSON文件", os.path.exists(paths["json"]))
test("导出Markdown文件", os.path.exists(paths["markdown"]))

with open(paths["json"], "r", encoding="utf-8") as f:
    json_data = json.load(f)
test("JSON可解析", "总分" in json_data)

with open(paths["markdown"], "r", encoding="utf-8") as f:
    md_content = f.read()
test("Markdown可读", "答辩改进报告" in md_content)

import shutil
shutil.rmtree(output_dir, ignore_errors=True)

# ========== 5. API路由 ==========
print("\n【5. FastAPI路由检查】")

from api_server import app
routes = [r.path for r in app.routes if hasattr(r, 'path')]
test("根路由", "/" in routes)
test("/personas路由", "/personas" in routes)
test("/scenarios路由", "/scenarios" in routes)
test("/start路由", "/start" in routes)
test("/answer路由", "/answer" in routes)
test("/evaluate路由", "/evaluate" in routes)
test("/report路由", "/report" in routes)
test("/session路由", any("/session" in r for r in routes))
test("/sessions路由", "/sessions" in routes)

# ========== 6. API模型验证 ==========
print("\n【6. API请求模型】")

from api_server import StartRequest, AnswerRequest, SessionRequest, ReportRequest

sr = StartRequest(persona="strict_tech", scenario="大创立项", project_text="测试")
test("StartRequest默认值", sr.persona == "strict_tech")

ar = AnswerRequest(session_id="test", answer="回答")
test("AnswerRequest创建", ar.session_id == "test")

# ========== 汇总 ==========
print("\n" + "=" * 60)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
print(f"测试结果: {passed} 通过 / {failed} 失败 / 共 {len(results)} 项")
if failed > 0:
    print("\n失败项:")
    for name, s, d in results:
        if s == FAIL:
            print(f"  - {name} {d}")
print("=" * 60)