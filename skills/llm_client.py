"""
LLM统一调用层
兼容智谱/通义/DeepSeek/OpenAI等OpenAI兼容协议API。
支持：单轮对话、多轮对话、流式输出、对话历史管理。
"""

import json
import os
from typing import Optional, Generator
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Message:
    """对话消息"""
    role: str  # system / user / assistant
    content: str


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""


class ConversationHistory:
    """对话历史管理器"""

    def __init__(self, system_prompt: str = "", max_turns: int = 20):
        self._messages: list = []
        self._max_turns = max_turns
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))
        self._trim()

    def get_messages(self) -> list:
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def _trim(self) -> None:
        system_msgs = [m for m in self._messages if m.role == "system"]
        non_system = [m for m in self._messages if m.role != "system"]
        if len(non_system) > self._max_turns * 2:
            non_system = non_system[-(self._max_turns * 2):]
        self._messages = system_msgs + non_system

    def last_assistant(self) -> str:
        for m in reversed(self._messages):
            if m.role == "assistant":
                return m.content
        return ""

    def clear(self) -> None:
        system_msgs = [m for m in self._messages if m.role == "system"]
        self._messages = system_msgs

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self._messages if m.role == "user")

    def to_dict(self) -> list:
        return [{"role": m.role, "content": m.content} for m in self._messages]


class LLMClient:
    """
    LLM统一调用客户端。
    所有主流国产大模型均兼容OpenAI协议，只需切换base_url和model。
    """

    PRESETS = {
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "model": "glm-4-flash",
        },
        "zhipu-plus": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "model": "glm-4-plus",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        preset: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60,
    ):
        if preset and preset in self.PRESETS:
            p = self.PRESETS[preset]
            self.base_url = base_url or p["base_url"]
            self.model = model or p["model"]
        else:
            self.base_url = base_url or "https://open.bigmodel.cn/api/paas/v4"
            self.model = model or "glm-4-flash"

        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        单次对话调用。

        Args:
            messages: 消息列表 [{"role": "...", "content": "..."}]
            model: 覆盖默认模型
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大token

        Returns:
            LLMResponse
        """
        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
        )

    def chat_stream(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        流式对话调用，逐token返回。

        Args:
            messages: 消息列表
            model: 覆盖默认模型
            temperature: 覆盖默认温度

        Yields:
            每个token片段
        """
        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            },
            timeout=self.timeout,
            stream=True,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    yield delta["content"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def simple_chat(self, prompt: str, system_prompt: str = "") -> str:
        """
        简易对话：单条prompt → 单条回复。

        Args:
            prompt: 用户输入
            system_prompt: 系统提示

        Returns:
            助手回复文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = self.chat(messages)
        return resp.content

    @property
    def info(self) -> dict:
        """客户端配置信息"""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_set": bool(self.api_key),
        }