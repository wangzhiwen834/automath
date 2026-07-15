"""LLM 抽象层：统一接口，底层适配 OpenAI 兼容(DeepSeek/千问/GLM) 与 Anthropic(Claude)。

设计要点：
- 所有 provider 实现统一的 chat() / stream() 接口
- 国产三家接口都是 OpenAI 兼容，复用一个实现，靠 base_url 区分
- 流式输出统一成"逐 token yield 字符串"，上层不关心来源
- 通过 config.yaml 的 model_registry 注册，运行时按 key 切换
"""
from __future__ import annotations

from typing import Iterator, Literal

import httpx
from openai import OpenAI
from anthropic import Anthropic

from app.config import ModelConfig, get_settings

# 消息格式统一为 OpenAI 风格
Role = Literal["system", "user", "assistant"]


class Message(dict):
    """简单的消息类型，兼容 OpenAI 格式: {"role": ..., "content": ...}。"""

    def __init__(self, role: Role, content: str) -> None:
        super().__init__(role=role, content=content)


class LLMProvider:
    """统一的 LLM 接口。所有 provider 实现这两个方法。"""

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """同步对话，返回完整文本。"""
        raise NotImplementedError

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """流式对话，逐块 yield 文本片段。"""
        raise NotImplementedError


# ====================================================================
# OpenAI 兼容 Provider：适配 DeepSeek / 通义千问 / GLM
# ====================================================================
class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        # read 超时 120s：流式时只要持续来 token 就不会触发；
        # 卡死（无数据）120s 后抛 ReadTimeout，由编排器标记失败而非无限挂起。
        self.client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
            max_retries=2,
        )

    def _kwargs(self, temperature, max_tokens) -> dict:
        return {
            "model": self.cfg.model,
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }

    def chat(self, messages, *, temperature=None, max_tokens=None) -> str:
        resp = self.client.chat.completions.create(
            messages=[dict(m) for m in messages],
            stream=False,
            **self._kwargs(temperature, max_tokens),
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            messages=[dict(m) for m in messages],
            stream=True,
            **self._kwargs(temperature, max_tokens),
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ====================================================================
# Anthropic Provider：适配 Claude
# ====================================================================
class AnthropicProvider(LLMProvider):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.client = Anthropic(
            api_key=cfg.api_key,
            timeout=httpx.Timeout(120.0, connect=10.0),
            max_retries=2,
        )

    def _split_system(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Anthropic 的 system 消息要单独传，从 messages 里抽出来。"""
        system = None
        conv: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conv.append(dict(m))
        return system, conv

    def chat(self, messages, *, temperature=None, max_tokens=None) -> str:
        system, conv = self._split_system(messages)
        resp = self.client.messages.create(
            model=self.cfg.model,
            system=system or "",
            messages=conv,
            temperature=temperature if temperature is not None else self.cfg.temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,
        )
        return resp.content[0].text if resp.content else ""

    def stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        system, conv = self._split_system(messages)
        with self.client.messages.stream(
            model=self.cfg.model,
            system=system or "",
            messages=conv,
            temperature=temperature if temperature is not None else self.cfg.temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text


# ====================================================================
# 工厂：按 config.yaml 的 provider 字段实例化
# ====================================================================
_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai-compatible": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
}

# 简单缓存，避免重复建客户端
_cache: dict[str, LLMProvider] = {}


def get_llm(model_key: str | None = None) -> LLMProvider:
    """获取指定模型的 LLM 实例（带缓存）。

    model_key 为空时用 config.yaml 的 default_model。
    """
    cfg = get_settings().get_model(model_key)
    if cfg.key in _cache:
        return _cache[cfg.key]

    provider_cls = _PROVIDERS.get(cfg.provider)
    if provider_cls is None:
        raise ValueError(
            f"未知的 provider 类型 '{cfg.provider}'（模型 {cfg.key}）。"
            f"支持: {list(_PROVIDERS.keys())}"
        )

    instance = provider_cls(cfg)
    _cache[cfg.key] = instance
    return instance
