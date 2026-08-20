"""
LLM调用层 + 评委引擎 自动化测试
离线测试：验证数据结构、人设配置、Session管理、Prompt构建等
在线测试：标记为[需API]，需配置LLM_API_KEY后取消注释运行
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from skills.llm_client import LLMClient, ConversationHistory, Message, LLMResponse
from skills.judge_engine import (
    JudgeEngine, JudgePersona, DefenseSession, DefenseTurn,
    EvaluationResult, DimensionScore,
    JUDGE_PRESETS, EVALUATION_CRITERIA,
)

PASS = "✓ 通过"
FAIL = "✗ 失败"
results = []


def test(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


print("=" * 60)
print("LLM调用层 + 评委引擎 自动化测试")
print("=" * 60)

# ========== 1. LLMClient ==========
print("\n【1. LLMClient 配置与预设】")

client = LLMClient(api_key="test_key", preset="zhipu")
test("智谱预设base_url", "bigmodel" in client.base_url)
test("智谱预设model", "glm" in client.model)
test("api_key已设置", client.api_key == "test_key")

client_ds = LLMClient(api_key="test_key", preset="deepseek")
test("DeepSeek预设", "deepseek" in client_ds.base_url)

client_qwen = LLMClient(api_key="test_key", preset="qwen")
test("通义预设", "dashscope" in client_qwen.base_url or "aliyuncs" in client_qwen.base_url)

client_openai = LLMClient(api_key="test_key", preset="openai")
test("OpenAI预设", "openai.com" in client_openai.base_url)

client_custom = LLMClient(api_key="k", base_url="https://custom.api/v1", model="my-model")
test("自定义配置", client_custom.model == "my-model" and "custom" in client_custom.base_url)

info = client.info
test("info字典完整", all(k in info for k in ["base_url", "model", "temperature", "api_key_set"]))

# ========== 2. ConversationHistory ==========
print("\n【2. ConversationHistory 对话历史管理】")

history = ConversationHistory(system_prompt="你是评委", max_turns=5)
test("初始system_prompt", len(history._messages) == 1 and history._messages[0].role == "system")

history.add("user", "你好")
history.add("assistant", "你好，请问有什么问题？")
test("添加消息", len(history._messages) == 3)
test("turn_count", history.turn_count == 1)

msgs = history.get_messages()
test("get_messages格式", msgs[0]["role"] == "system" and msgs[1]["role"] == "user")

last = history.last_assistant()
test("last_assistant", "你好" in last)

history.clear()
test("clear保留system", len(history._messages) == 1 and history._messages[0].role == "system")

# 测试裁剪
history2 = ConversationHistory(max_turns=2)
for i in range(10):
    history2.add("user", f"q{i}")
    history2.add("assistant", f"a{i}")
test("历史裁剪", len([m for m in history2._messages if m.role != "system"]) <= 4)

# ========== 3. 评委人设 ==========
print("\n【3. 评委人设 JUDGE_PRESETS】")

test("5种人设", len(JUDGE_PRESETS) == 5)
for key, persona in JUDGE_PRESETS.items():
    test(f"人设[{key}]有system_prompt", len(persona.system_prompt) > 50)
    test(f"人设[{key}]有scoring_weights", len(persona.scoring_weights) > 0)
    weight_sum = sum(persona.scoring_weights.values())
    test(f"人设[{key}]权重和≈1", abs(weight_sum - 1.0) < 0.01, f"{weight_sum:.2f}")

# ========== 4. 评审标准 ==========
print("\n【4. 评审标准 EVALUATION_CRITERIA】")

test("大创立项标准", "大创立项" in EVALUATION_CRITERIA)
test("竞赛路演标准", "竞赛路演" in EVALUATION_CRITERIA)
for scenario, criteria in EVALUATION_CRITERIA.items():
    test(f"{scenario}有5个维度", len(criteria) == 5)

# ========== 5. DefenseSession & DefenseTurn ==========
print("\n【5. DefenseSession & DefenseTurn 数据结构】")

turn = DefenseTurn(turn_id=1, question="你的创新点是什么？")
turn.answer = "我们的创新点是跨模态对齐"
test("DefenseTurn创建", turn.turn_id == 1 and turn.answer != "")

session = DefenseSession(
    session_id="test_001",
    persona=JUDGE_PRESETS["strict_tech"],
    scenario="大创立项",
)
test("DefenseSession创建", session.session_id == "test_001")
test("Session人设", "严厉" in session.persona.name)
test("Session无turns", len(session.turns) == 0)

session.turns.append(turn)
d = session.to_dict()
test("to_dict序列化", "turns" in d and len(d["turns"]) == 1)
test("to_dict含人设", "严厉" in d["persona"])

# ========== 6. EvaluationResult ==========
print("\n【6. EvaluationResult 评审结果】")

eval_result = EvaluationResult(
    scores=[
        DimensionScore("技术深度", 85, "技术方案有深度"),
        DimensionScore("创新性", 78, "创新点明确"),
        DimensionScore("可行性", 72, "部分技术风险"),
        DimensionScore("表达清晰度", 80, "表达较清晰"),
        DimensionScore("应用价值", 75, "有一定应用前景"),
    ],
    total_score=78.6,
    strengths=["技术方案扎实", "创新点明确"],
    weaknesses=["部分技术风险未充分讨论"],
    suggestions=["补充技术风险分析", "增加对比实验"],
)
test("EvaluationResult创建", len(eval_result.scores) == 5)
test("总分", eval_result.total_score > 0)

summary = eval_result.summary()
test("summary含总分", "78.6" in summary)
test("summary含维度", "技术深度" in summary)
test("summary含建议", "补充技术风险分析" in summary)

ed = eval_result.to_dict()
test("to_dict完整", "各维度评分" in ed and "总分" in ed and "优势" in ed)

# ========== 7. JudgeEngine 离线功能 ==========
print("\n【7. JudgeEngine 离线功能】")

mock_client = LLMClient(api_key="mock", preset="zhipu")
engine = JudgeEngine(llm_client=mock_client, max_followups=2, max_turns=8)
test("JudgeEngine创建", engine.max_followups == 2)

personas = JudgeEngine.list_personas()
test("列出人设", len(personas) == 5)
test("人设含strict_tech", "strict_tech" in personas)

scenarios = JudgeEngine.list_scenarios()
test("列出场景", len(scenarios) == 2)

# start_session
session2 = engine.start_session(persona="strict_tech", scenario="大创立项")
test("start_session", session2.session_id.startswith("defense_"))
test("session人设正确", session2.persona.style == "strict_tech")
test("session在引擎中", session2.session_id in engine._sessions)

# 错误人设
try:
    engine.start_session(persona="nonexistent")
    test("错误人设报错", False)
except ValueError:
    test("错误人设报错", True)

# get_session
retrieved = engine.get_session(session2.session_id)
test("get_session", retrieved is session2)

# list_sessions
test("list_sessions", session2.session_id in engine.list_sessions())

# export_session
export_path = os.path.join(os.path.dirname(__file__), "_test_session.json")
engine.export_session(session2, export_path)
test("export_session文件存在", os.path.exists(export_path))
with open(export_path, "r", encoding="utf-8") as f:
    exported = json.load(f)
test("导出JSON有效", "session_id" in exported and "persona" in exported)
os.remove(export_path)

# ========== 8. 带RAG的Session ==========
print("\n【8. 带RAG的Session】")

from skills.rag_engine import RAGEngine

rag = RAGEngine(chunk_size=300)
sample = """
项目背景：基于AI知识图谱与多模态视觉技术构建智能问答系统。
创新点：提出跨模态知识对齐方法，融合视觉与语义信息。
技术路线：采用CLIP模型进行图文对齐，结合GNN进行图谱推理。
"""
rag.index_text(sample, source="test", strategy="section")

engine_with_rag = JudgeEngine(llm_client=mock_client, rag_engine=rag)
session3 = engine_with_rag.start_session(persona="encouraging", project_text=sample)
test("带RAG的session", session3.persona.style == "encouraging")
test("RAG已索引", rag.store.size > 0)

# ========== 9. LLMClient预设完整性 ==========
print("\n【9. LLMClient预设完整性】")

for preset_name in LLMClient.PRESETS:
    p = LLMClient.PRESETS[preset_name]
    test(f"预设[{preset_name}]有base_url", bool(p["base_url"]))
    test(f"预设[{preset_name}]有model", bool(p["model"]))

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

print("""
========================================
[需API] 在线测试说明
========================================
配置环境变量后可运行完整答辩流程:

  set LLM_API_KEY=你的API密钥

然后在Python中:

  from skills.llm_client import LLMClient
  from skills.rag_engine import RAGEngine
  from skills.judge_engine import JudgeEngine

  client = LLMClient(preset="zhipu")  # 或 deepseek/qwen
  rag = RAGEngine()
  rag.index_document("申报书.docx")

  engine = JudgeEngine(llm_client=client, rag_engine=rag)
  session = engine.start_session(persona="strict_tech")

  q1 = engine.ask_first_question(session)
  print(f"评委: {q1}")

  engine.answer(session, "我们的创新点是...")
  fq = engine.followup(session)
  if fq:
      print(f"追问: {fq}")
      engine.answer(session, "具体来说...")

  result = engine.evaluate(session)
  print(result.summary())
========================================
""")