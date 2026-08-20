"""
FastAPI后端API
为前端提供答辩模拟的完整REST接口。
启动: uvicorn api_server:app --reload --port 8000
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from skills.llm_client import LLMClient
from skills.rag_engine import RAGEngine
from skills.judge_engine import JudgeEngine, JUDGE_PRESETS, EVALUATION_CRITERIA
from skills.report_generator import ReportGenerator
from skills.doc_parser import DocParserSkill
from skills.teacher_assistant import TeacherAssistant
from skills.digital_human import DigitalHumanSkill

app = FastAPI(
    title="AI模拟答辩智能体 API",
    description="基于大语言模型与RAG的答辩模拟系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/app")
def serve_app():
    """提供前端页面"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

_engines: dict = {}


def _get_engine(api_key: str = "", preset: str = "") -> tuple:
    cache_key = f"{preset}:{api_key[:8]}"
    if cache_key in _engines:
        return _engines[cache_key]

    key = api_key or os.getenv("LLM_API_KEY", "")
    p = preset or os.getenv("LLM_PRESET", "qwen")
    client = LLMClient(api_key=key, preset=p)
    rag = RAGEngine(chunk_size=400)
    judge = JudgeEngine(llm_client=client, rag_engine=rag, max_followups=2, max_turns=8)
    reporter = ReportGenerator(llm_client=client, rag_engine=rag)
    _engines[cache_key] = (judge, reporter, rag)
    return judge, reporter, rag


# ========== 请求/响应模型 ==========

class StartRequest(BaseModel):
    persona: str = "strict_tech"
    scenario: str = "大创立项"
    project_text: str = ""
    max_turns: int = 5
    api_key: str = ""
    preset: str = ""


@app.post("/upload")
async def upload_document(file: UploadFile = None):
    """上传docx/pdf文件，返回提取的文本"""
    if not file or not file.filename:
        raise HTTPException(400, "请上传文件")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".docx", ".pdf"}:
        raise HTTPException(400, f"仅支持docx和pdf格式，收到: {ext}。如果是.doc旧格式请另存为.docx")

    content = await file.read()
    if not content:
        raise HTTPException(400, "文件内容为空")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        parser = DocParserSkill()
        text = parser.parse(tmp_path)
        if not text.strip():
            raise HTTPException(500, "未提取到文本，可能是扫描版PDF（图片格式），请用文字版PDF或直接粘贴文本")
        return {"filename": file.filename, "text": text, "char_count": len(text)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"解析失败: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

class AnswerRequest(BaseModel):
    session_id: str
    answer: str
    api_key: str = ""
    preset: str = ""

class SessionRequest(BaseModel):
    session_id: str
    api_key: str = ""
    preset: str = ""

class ReportRequest(BaseModel):
    session_id: str
    api_key: str = ""
    preset: str = ""


# ========== API路由 ==========

@app.get("/")
def root():
    return {"name": "AI模拟答辩智能体", "version": "1.0.0", "status": "running"}


@app.get("/personas")
def list_personas():
    """列出所有可用评委人设"""
    return JudgeEngine.list_personas()


@app.get("/scenarios")
def list_scenarios():
    """列出所有答辩场景"""
    return JudgeEngine.list_scenarios()


@app.post("/start")
def start_session(req: StartRequest):
    """开始答辩会话，评委提出第一个问题"""
    judge, _, rag = _get_engine(req.api_key, req.preset)

    if req.persona not in JUDGE_PRESETS:
        raise HTTPException(400, f"未知评委人设: {req.persona}，可选: {list(JUDGE_PRESETS.keys())}")

    judge.max_turns = req.max_turns

    try:
        session = judge.start_session(
            persona=req.persona,
            scenario=req.scenario,
            project_text=req.project_text,
        )
        question = judge.ask_first_question(session)

        return {
            "session_id": session.session_id,
            "persona": session.persona.name,
            "scenario": session.scenario,
            "question": question,
            "turn": 1,
            "total_turns": req.max_turns,
        }
    except Exception as e:
        raise HTTPException(500, f"启动失败: {str(e)}")


@app.post("/answer")
def submit_answer(req: AnswerRequest):
    """学生回答，评委决定追问或提出新问题"""
    judge, _, _ = _get_engine(req.api_key, req.preset)

    session = judge.get_session(req.session_id)
    if not session:
        raise HTTPException(404, f"会话不存在: {req.session_id}")

    try:
        judge.answer(session, req.answer)

        followup = judge.followup(session)

        if followup:
            return {
                "session_id": session.session_id,
                "type": "followup",
                "question": followup,
                "turn": session.turns[-1].turn_id,
                "is_followup": True,
            }

        next_q = judge.next_question(session)
        if next_q:
            return {
                "session_id": session.session_id,
                "type": "new_question",
                "question": next_q,
                "turn": session.turns[-1].turn_id,
                "is_followup": False,
            }

        return {
            "session_id": session.session_id,
            "type": "finished",
            "question": None,
            "turn": len(session.turns),
            "is_followup": False,
            "message": "答辩结束，可以调用 /evaluate 获取评审结果",
        }
    except Exception as e:
        raise HTTPException(500, f"回答处理失败: {str(e)}")


@app.post("/evaluate")
def evaluate_session(req: SessionRequest):
    """答辩结束，评审打分"""
    judge, _, _ = _get_engine(req.api_key, req.preset)

    session = judge.get_session(req.session_id)
    if not session:
        raise HTTPException(404, f"会话不存在: {req.session_id}")

    try:
        result = judge.evaluate(session)
        return {
            "session_id": session.session_id,
            "evaluation": result.to_dict(),
        }
    except Exception as e:
        raise HTTPException(500, f"评审失败: {str(e)}")


@app.post("/report")
def generate_report(req: ReportRequest):
    """生成完整答辩改进报告"""
    judge, reporter, _ = _get_engine(req.api_key, req.preset)

    session = judge.get_session(req.session_id)
    if not session:
        raise HTTPException(404, f"会话不存在: {req.session_id}")

    if not session.evaluation:
        try:
            judge.evaluate(session)
        except Exception as e:
            raise HTTPException(500, f"评审失败: {str(e)}")

    try:
        report = reporter.generate(session)
        return report.to_dict()
    except Exception as e:
        raise HTTPException(500, f"报告生成失败: {str(e)}")


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """获取答辩会话详情"""
    for cache_key, (judge, _, _) in _engines.items():
        session = judge.get_session(session_id)
        if session:
            return session.to_dict()
    raise HTTPException(404, f"会话不存在: {session_id}")


@app.get("/sessions")
def list_sessions():
    """列出所有活跃会话"""
    all_sessions = []
    for cache_key, (judge, _, _) in _engines.items():
        for sid in judge.list_sessions():
            session = judge.get_session(sid)
            if session:
                all_sessions.append({
                    "session_id": sid,
                    "persona": session.persona.name if session.persona else "",
                    "scenario": session.scenario,
                    "turns": len(session.turns),
                    "evaluated": session.evaluation is not None,
                })
    return all_sessions


# ========== 教师端接口 ==========

_teachers: dict = {}


class TeacherAnalyzeRequest(BaseModel):
    project_text: str = ""
    scenario: str = "大创立项"
    api_key: str = ""
    preset: str = ""


class TeacherQuestionSetRequest(BaseModel):
    project_text: str = ""
    scenario: str = "大创立项"
    num_questions: int = 12
    api_key: str = ""
    preset: str = ""


def _get_teacher(api_key: str = "", preset: str = "") -> TeacherAssistant:
    cache_key = f"teacher:{preset}:{api_key[:8]}"
    if cache_key in _teachers:
        return _teachers[cache_key]
    key = api_key or os.getenv("LLM_API_KEY", "")
    p = preset or os.getenv("LLM_PRESET", "qwen")
    client = LLMClient(api_key=key, preset=p)
    teacher = TeacherAssistant(llm_client=client)
    _teachers[cache_key] = teacher
    return teacher


@app.get("/teacher")
def serve_teacher():
    """教师端页面"""
    return FileResponse(os.path.join(STATIC_DIR, "teacher.html"))


@app.post("/teacher/analyze")
def teacher_analyze(req: TeacherAnalyzeRequest):
    """教师端：分析项目材料"""
    if not req.project_text.strip():
        raise HTTPException(400, "请输入项目材料")
    teacher = _get_teacher(req.api_key, req.preset)
    try:
        analysis = teacher.analyze_text(req.project_text, scenario=req.scenario)
        return {
            "filename": analysis.filename,
            "char_count": analysis.char_count,
            "keywords_matched": analysis.keywords_matched,
            "criteria_scores": analysis.criteria_scores,
            "summary": analysis.summary,
        }
    except Exception as e:
        raise HTTPException(500, f"分析失败: {str(e)}")


@app.post("/teacher/question-set")
def teacher_question_set(req: TeacherQuestionSetRequest):
    """教师端：生成问题集"""
    if not req.project_text.strip():
        raise HTTPException(400, "请输入项目材料")
    teacher = _get_teacher(req.api_key, req.preset)
    try:
        items = teacher.generate_question_set(req.project_text, req.scenario, req.num_questions)
        return {
            "questions": [
                {
                    "question": q.question,
                    "reference_answer": q.reference_answer,
                    "category": q.category,
                    "difficulty": q.difficulty,
                    "scoring_points": q.scoring_points,
                    "source_section": q.source_section,
                }
                for q in items
            ],
            "total": len(items),
        }
    except Exception as e:
        raise HTTPException(500, f"生成失败: {str(e)}")


# ========== 视频答辩 + 数字人接口 ==========

class AvatarRequest(BaseModel):
    text: str
    avatar_name: str = "male_teacher"
    api_key: str = ""


@app.get("/video")
def serve_video():
    """视频答辩页面"""
    return FileResponse(os.path.join(STATIC_DIR, "video.html"))


@app.post("/avatar/check")
def check_avatar():
    """检查D-ID数字人是否可用"""
    dh = DigitalHumanSkill()
    return {"available": dh.available, "message": "D-ID已配置" if dh.available else "未配置DID_API_KEY，使用CSS动画数字人(免费)"}


@app.post("/avatar/talk")
def avatar_talk(req: AvatarRequest):
    """生成数字人说话视频"""
    dh = DigitalHumanSkill(api_key=req.api_key)
    if not dh.available:
        raise HTTPException(400, "未配置DID_API_KEY，请先在.env中设置")
    try:
        result = dh.create_talk(req.text, avatar_name=req.avatar_name, wait=True, timeout=60)
        return result
    except Exception as e:
        raise HTTPException(500, f"数字人生成失败: {str(e)}")


# ========== 语音识别接口（后端STT） ==========

@app.post("/stt")
async def speech_to_speech(file: UploadFile = None):
    """
    语音转文字：接收音频文件，用通义千问qwen-audio-turbo识别。
    解决Chrome Web Speech API在大陆无法连接Google服务器的问题。
    """
    if not file or not file.filename:
        raise HTTPException(400, "请上传音频文件")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "音频内容为空")

    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        raise HTTPException(500, "未配置LLM_API_KEY")

    import base64
    import requests as req_lib

    audio_b64 = base64.b64encode(audio_bytes).decode()
    mime = file.content_type or "audio/webm"
    data_url = f"data:{mime};base64,{audio_b64}"

    try:
        resp = req_lib.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "qwen-audio-turbo",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请将这段语音转写为纯文字，只输出转写结果，不要任何标点以外的额外内容"},
                        {"type": "audio", "audio_url": data_url},
                    ],
                }],
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"语音识别API错误({resp.status_code}): {resp.text[:300]}")

        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return {"text": text, "model": "qwen-audio-turbo"}
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(500, f"语音识别响应格式异常: {resp.text[:300]}")
    except Exception as e:
        raise HTTPException(500, f"语音识别失败: {str(e)}")