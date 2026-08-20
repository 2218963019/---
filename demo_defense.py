"""
完整答辩模拟演示
使用阿里云通义千问API，模拟一次真实答辩流程
"""

from skills.llm_client import LLMClient
from skills.rag_engine import RAGEngine
from skills.judge_engine import JudgeEngine

PROJECT_TEXT = """
项目名称：基于知识图谱与多模态视觉的智能问答系统

项目背景：传统问答系统仅能处理文本输入，无法理解图像、图表等视觉信息。
本项目基于AI知识图谱与多模态视觉技术，构建支持图文混合输入的智能问答系统，
解决科研、教育场景中"看图提问"的需求痛点。

研究目标：
1. 构建领域知识图谱，覆盖计算机科学核心概念及关系
2. 实现跨模态知识对齐，将视觉特征映射到图谱语义空间
3. 支持图文混合输入的智能问答，准确率超过90%

创新点：
1. 提出跨模态知识对齐方法（Cross-Modal Knowledge Alignment），将CLIP视觉编码与知识图谱嵌入对齐到统一语义空间
2. 设计图注意力引导的多模态融合机制（GAT-Guided Fusion），利用图谱结构信息指导视觉-文本特征交互
3. 实现基于检索增强的推理链生成（RAG-Chain Reasoning），结合知识图谱路径检索与大模型推理

技术路线：
- 视觉编码：CLIP ViT-L/14 提取图像特征
- 图谱构建：基于SciERC数据集，使用BERT+CRF抽取实体关系，存入Neo4j
- 多模态融合：Cross-Attention + 图注意力网络(GAT) 双路径融合
- 推理生成：RAG检索图谱子图 → 拼接视觉特征 → LLM生成答案
- 部署：FastAPI后端 + Vue3前端 + Triton推理服务

实验评估：
- 自建MM-QA数据集（2000个图文问答对）
- 准确率92.3%，F1分数89.7%，较基线TextQA提升15.2%
- 推理延迟：单次问答平均380ms（含检索+推理）

系统架构：
- 前端：Vue3 + TypeScript + ECharts图谱可视化
- 后端：FastAPI + WebSocket流式输出
- 知识图谱：Neo4j + 自建CS-KG（1.2万实体，3.5万关系）
- 向量检索：Milvus（存储CLIP嵌入）
- 模型服务：Triton Inference Server

团队：5人，1名研究生负责算法，2名本科生负责工程，1名负责数据，1名负责前端

局限性：
1. 仅支持图文两种模态，视频和音频理解待扩展
2. 知识图谱仅覆盖计算机科学领域，跨领域泛化能力有限
3. 推理延迟380ms，实时性有待优化

未来规划：
1. 扩展视频模态，接入Video-LLaVA模型
2. 引入模型蒸馏（Qwen2.5-0.5B），降低推理延迟至100ms以内
3. 增加用户反馈闭环，持续优化图谱和模型
4. 探索跨领域迁移，构建多学科知识图谱
"""


def main():
    print("=" * 60)
    print("AI模拟答辩 完整演示")
    print("=" * 60)

    # 1. 初始化
    client = LLMClient(preset="qwen", temperature=0.7)
    rag = RAGEngine(chunk_size=400)
    rag.index_text(PROJECT_TEXT, source="申报书", strategy="section")
    print(f"\nRAG索引完成: {rag.stats}")

    engine = JudgeEngine(llm_client=client, rag_engine=rag, max_followups=1, max_turns=3)

    # 2. 选择评委
    personas = JudgeEngine.list_personas()
    print(f"\n可用评委: {personas}")

    session = engine.start_session(persona="strict_tech", scenario="大创立项")
    print(f"答辩会话: {session.session_id}")
    print(f"评委: {session.persona.name}")

    # 3. 第一轮
    print("\n" + "─" * 50)
    print("【第1轮】")
    q1 = engine.ask_first_question(session)
    print(f"评委: {q1}")

    a1 = "我们的核心创新点是跨模态知识对齐方法。传统方法将视觉和文本分开处理，我们通过将CLIP视觉编码与知识图谱嵌入对齐到统一语义空间，实现了图文联合推理。具体来说，我们设计了GAT-Guided Fusion机制，利用图谱的注意力权重来指导视觉-文本特征的交互融合。"
    print(f"\n学生: {a1}")
    engine.answer(session, a1)

    fq1 = engine.followup(session)
    if fq1:
        print(f"\n评委追问: {fq1}")
        fa1 = "我们在MM-QA数据集上做了消融实验，去掉GAT引导后F1下降了6.3个百分点，说明图谱结构信息对融合确实有显著指导作用。对齐训练使用了对比学习损失，batch size 256，训练了50个epoch。"
        print(f"\n学生: {fa1}")
        engine.answer(session, fa1)

    # 4. 第二轮
    print("\n" + "─" * 50)
    print("【第2轮】")
    q2 = engine.next_question(session)
    if q2:
        print(f"评委: {q2}")

        a2 = "推理延迟380ms主要由三部分组成：向量检索约80ms，图谱路径搜索约60ms，LLM生成约240ms。优化方向是引入模型蒸馏，用Qwen2.5-0.5B替代当前大模型做推理，预计可以降到100ms以内。同时我们也在探索缓存热门查询的图谱子图来减少检索时间。"
        print(f"\n学生: {a2}")
        engine.answer(session, a2)

        fq2 = engine.followup(session)
        if fq2:
            print(f"\n评委追问: {fq2}")
            fa2 = "蒸馏后的0.5B模型在MM-QA上准确率下降了约3个百分点，从92.3%降到89.1%，但延迟从240ms降到35ms，性价比很高。我们考虑用蒸馏模型做快速初筛，大模型做精排的两阶段方案。"
            print(f"\n学生: {fa2}")
            engine.answer(session, fa2)

    # 5. 第三轮
    print("\n" + "─" * 50)
    print("【第3轮】")
    q3 = engine.next_question(session)
    if q3:
        print(f"评委: {q3}")

        a3 = "知识图谱目前覆盖计算机科学领域，包含1.2万实体和3.5万关系，基于SciERC数据集构建。跨领域泛化是我们的局限之一，未来计划通过迁移学习将图谱扩展到物理、数学等理工科领域，预计每个新领域需要额外标注约2000个实体关系对。"
        print(f"\n学生: {a3}")
        engine.answer(session, a3)

    # 6. 评审
    print("\n" + "─" * 50)
    print("【答辩结束，评审中...】")
    result = engine.evaluate(session)
    print(result.summary())

    # 7. 导出
    engine.export_session(session, "defense_result.json")
    print(f"\n答辩记录已导出: defense_result.json")


if __name__ == "__main__":
    main()