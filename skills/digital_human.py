"""
D-ID数字人Skill
接入D-ID API，将文本转为数字人说话视频（嘴型同步+表情自然）。
注册地址: https://studio.d-id.com
"""

import os
import time
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class DigitalHumanSkill:
    """
    D-ID数字人Skill。
    输入文本 → 调用D-ID API → 返回数字人说话视频URL。

    使用前:
        1. 注册 https://studio.d-id.com 获取API Key
        2. 在.env中添加: DID_API_KEY=你的key
    """

    D_ID_API = "https://api.d-id.com"

    PRESET_AVATARS = {
        "male_teacher": "https://create-images-results.d-id.com/Default/male_teacher.jpg",
        "female_teacher": "https://create-images-results.d-id.com/Default/female_teacher.jpg",
        "strict_reviewer": "https://create-images-results.d-id.com/Default/strict_reviewer.jpg",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DID_API_KEY", "")
        self.base_url = self.D_ID_API

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def create_talk(
        self,
        text: str,
        avatar_url: str = "",
        avatar_name: str = "male_teacher",
        voice_lang: str = "zh-CN",
        voice_provider: str = "microsoft",
        wait: bool = True,
        timeout: int = 60,
    ) -> dict:
        """
        生成数字人说话视频。

        Args:
            text: 要说的文本
            avatar_url: 自定义头像图片URL（优先）
            avatar_name: 预设头像名(male_teacher/female_teacher/strict_reviewer)
            voice_lang: 语音语言
            voice_provider: 语音提供商(microsoft/elevenLabs/amazon)
            wait: 是否等待视频生成完成
            timeout: 等待超时秒数

        Returns:
            {"talk_id": "...", "result_url": "视频URL"} 或 {"talk_id": "...", "status": "started"}
        """
        if not self.available:
            raise RuntimeError("未配置D-ID API Key，请在.env中设置DID_API_KEY")

        source_url = avatar_url or self.PRESET_AVATARS.get(avatar_name, self.PRESET_AVATARS["male_teacher"])

        resp = requests.post(
            f"{self.base_url}/talks",
            headers={
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "source_url": source_url,
                "script": {
                    "type": "text",
                    "input": text,
                    "provider": {
                        "type": voice_provider,
                        "voice_id": self._get_voice_id(voice_provider, voice_lang),
                    },
                },
                "config": {
                    "stitch": True,
                    "sharpen": True,
                    "fluent": True,
                    "padAudio": 0.5,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        talk_id = data.get("id")

        if not wait:
            return {"talk_id": talk_id, "status": "started"}

        result = self._wait_for_result(talk_id, timeout)
        return result

    def _get_voice_id(self, provider: str, lang: str) -> str:
        """获取语音ID"""
        voices = {
            "microsoft": {
                "zh-CN": "zh-CN-YunxiNeural",
                "zh-TW": "zh-TW-HsiaoChenNeural",
                "en-US": "en-US-ChristopherNeural",
            },
            "elevenLabs": {
                "zh-CN": "eleven-labs-default",
                "en-US": "eleven-labs-default",
            },
        }
        return voices.get(provider, {}).get(lang, "zh-CN-YunxiNeural")

    def _wait_for_result(self, talk_id: str, timeout: int) -> dict:
        """轮询等待视频生成完成"""
        start = time.time()
        while time.time() - start < timeout:
            resp = requests.get(
                f"{self.base_url}/talks/{talk_id}",
                headers={"Authorization": f"Basic {self.api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")

            if status == "done":
                return {
                    "talk_id": talk_id,
                    "status": "done",
                    "result_url": data.get("result_url", ""),
                    "duration": data.get("duration", 0),
                }
            elif status == "error":
                raise RuntimeError(f"D-ID生成失败: {data.get('error', '未知错误')}")

            time.sleep(2)

        return {"talk_id": talk_id, "status": "timeout", "result_url": ""}

    def create_talk_stream(
        self,
        text: str,
        avatar_url: str = "",
        avatar_name: str = "male_teacher",
    ) -> dict:
        """
        流式生成（低延迟，适合实时答辩）。
        需要先建立stream连接。

        Returns:
            stream连接信息
        """
        if not self.available:
            raise RuntimeError("未配置D-ID API Key")

        source_url = avatar_url or self.PRESET_AVATARS.get(avatar_name, self.PRESET_AVATARS["male_teacher"])

        resp = requests.post(
            f"{self.base_url}/streams",
            headers={
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "source_url": source_url,
                "config": {"stitch": True, "fluent": True},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def list_avatars(self) -> dict:
        """列出可用预设头像"""
        return self.PRESET_AVATARS.copy()