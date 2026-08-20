"""
Skills 调用示例
演示四个Skill的独立调用与组合使用流程
"""

from skills.doc_parser import DocParserSkill
from skills.question_generator import QuestionGeneratorSkill
from skills.solution_optimizer import SolutionOptimizerSkill
from skills.rag_engine import RAGEngine


def demo_doc_parser():
    """Skill1: 文档解析调用示例"""
    print("=" * 50)
    print("Skill1: 文档解析Skill 调用示例")
    print("=" * 50)

    parser = DocParserSkill()

    # 单文件解析
    # text = parser.parse("项目申报书.docx")
    # print(f"提取文本字数: {len(text)}")
    # print(f"元信息: {parser.get_metadata()}")

    # 按章节切分
    # sections = parser.get_sections(keywords=["摘要", "技术路线", "创新点", "预期成果"])
    # for k, v in sections.items():
    #     print(f"\n--- {k} ---\n{v[:200]}...")

    # 批量解析
    # results = DocParserSkill.batch_parse(["申报书.docx", "论文.pdf", "报告.docx"])
    # for fp, text in results.items():
    #     print(f"{fp}: {len(text)}字")

    print("（取消注释即可运行，需提供实际文件路径）")


def demo_question_generator():
    """Skill2: 答辩问题生成调用示例"""
    print("\n" + "=" * 50)
    print("Skill2: 答辩问题生成Skill 调用示例")
    print("=" * 50)

    sample_text = """
    项目背景：本项目基于AI知识图谱与多模态视觉技术，构建智能问答系统。
    研究目标：实现知识图谱驱动的多模态理解与推理。
    创新点：提出跨模态知识对齐方法，融合视觉与语义信息。
    技术路线：采用CLIP模型进行图文对齐，结合GNN进行图谱推理。
    算法设计：基于Transformer架构，引入Cross-Attention实现多模态融合。
    实验评估：在自建数据集上准确率达到92.3%，优于基线方法。
    系统架构：前端Vue3 + 后端FastAPI + 知识图谱Neo4j + 向量数据库Milvus。
    局限性：当前仅支持图文模态，视频理解有待扩展。
    未来规划：扩展视频模态支持，优化实时推理性能。
    """

    generator = QuestionGeneratorSkill()
    result = generator.generate(sample_text)

    print(result.summary())

    # 导出到文件
    # generator.export_questions("答辩问题.txt")

    # 获取字典格式
    # print(result.to_dict())


def demo_solution_optimizer():
    """Skill3: 方案优化调用示例"""
    print("\n" + "=" * 50)
    print("Skill3: 方案优化Skill 调用示例")
    print("=" * 50)

    sample_text = """
    本项目构建了基于知识图谱的多模态视觉问答系统。
    知识图谱采用Neo4j存储，实体关系通过NLP抽取。
    多模态部分使用CLIP模型进行图文对齐，图像特征通过ViT提取。
    系统部署为Web服务，提供REST API接口。
    前端使用Vue3实现交互式问答界面。
    实验表明多模态融合显著提升了问答准确率。
    """

    optimizer = SolutionOptimizerSkill()
    result = optimizer.optimize(sample_text)

    print(result.summary())

    # 导出到文件
    # optimizer.export_suggestions("优化建议.txt")

    # 获取字典格式
    # print(result.to_dict())


def demo_rag_engine():
    """Skill4: RAG检索引擎调用示例"""
    print("\n" + "=" * 50)
    print("Skill4: RAG检索引擎 调用示例")
    print("=" * 50)

    sample_text = """
    项目背景：本项目基于AI知识图谱与多模态视觉技术，构建智能问答系统。
    研究目标：实现知识图谱驱动的多模态理解与推理。
    创新点：提出跨模态知识对齐方法，融合视觉与语义信息。
    技术路线：采用CLIP模型进行图文对齐，结合GNN进行图谱推理。
    算法设计：基于Transformer架构，引入Cross-Attention实现多模态融合。
    实验评估：在自建数据集上准确率达到92.3%，优于基线方法。
    系统架构：前端Vue3 + 后端FastAPI + 知识图谱Neo4j + 向量数据库Milvus。
    局限性：当前仅支持图文模态，视频理解有待扩展。
    未来规划：扩展视频模态支持，优化实时推理性能。
    """

    engine = RAGEngine(chunk_size=300)

    # 索引文本（也可用 engine.index_document("申报书.docx") 直接索引文件）
    n = engine.index_text(sample_text, source="申报书", strategy="section")
    print(f"索引完成: {n}个chunk, 统计: {engine.stats}")

    # 语义检索
    results = engine.retrieve("项目的创新点是什么？", top_k=3)
    for r in results:
        print(f"  [相关度:{r.score:.3f}] {r.chunk.content[:60]}...")

    # 带上下文检索（可直接拼入LLM prompt）
    context = engine.retrieve_with_context("技术路线和算法设计", top_k=2)
    print(f"\n{context}")

    # 持久化
    # engine.save("rag_store.json")
    # engine2 = RAGEngine()
    # engine2.load("rag_store.json")


def demo_pipeline():
    """组合调用：文档解析 → RAG索引 → 问题生成 + 方案优化"""
    print("\n" + "=" * 50)
    print("组合调用示例: 解析 → RAG索引 → 问题生成 + 方案优化")
    print("=" * 50)

    # Step 1: 解析文档
    # parser = DocParserSkill()
    # text = parser.parse("大创项目申报书.docx")

    # Step 2: RAG索引（为后续LLM交互提供知识检索）
    # engine = RAGEngine()
    # engine.index_document("大创项目申报书.docx", strategy="section")

    # Step 3: 检索相关内容作为LLM上下文
    # context = engine.retrieve_with_context("项目的创新点", top_k=3)

    # Step 4: 生成答辩问题
    # generator = QuestionGeneratorSkill()
    # question_result = generator.generate(text)
    # print(question_result.summary())

    # Step 5: 生成优化建议
    # optimizer = SolutionOptimizerSkill()
    # optimization_result = optimizer.optimize(text)
    # print(optimization_result.summary())

    print("（取消注释即可运行完整流水线，需提供实际文件路径）")


if __name__ == "__main__":
    demo_doc_parser()
    demo_question_generator()
    demo_solution_optimizer()
    demo_rag_engine()
    demo_pipeline()