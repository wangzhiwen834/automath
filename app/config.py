"""配置加载：从 config.yaml + .env 读取全局配置。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# 项目根目录：app/config.py 的上上级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ModelConfig:
    """单个模型的配置。"""

    def __init__(self, key: str, data: dict[str, Any]):
        self.key = key
        self.provider: str = data["provider"]
        self.base_url: str | None = data.get("base_url")
        self.model: str = data["model"]
        self.api_key_env: str = data["api_key_env"]
        self.max_tokens: int = data.get("max_tokens", 8192)
        self.temperature: float = data.get("temperature", 0.7)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    @property
    def available(self) -> bool:
        """该模型是否可用（api_key 已配置）。"""
        return bool(self.api_key)


class Settings:
    """全局配置单例。"""

    _instance: "Settings | None" = None

    def __init__(self) -> None:
        # 加载 .env
        load_dotenv(PROJECT_ROOT / ".env")

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self._raw: dict[str, Any] = yaml.safe_load(f)

        self.workspace_dir: Path = self._resolve_workspace()
        self.default_model: str = self._raw.get("default_model", "")

        # 模型注册表
        self.model_registry: dict[str, ModelConfig] = {
            key: ModelConfig(key, data)
            for key, data in self._raw.get("model_registry", {}).items()
        }

        # 各 Agent 默认模型
        self.agent_models: dict[str, str] = self._raw.get("agent_models", {})
        self.solver_config: dict[str, Any] = self._raw.get("solver", {})
        self.reviewer_config: dict[str, Any] = self._raw.get("reviewer", {})
        self.writer_config: dict[str, Any] = self._raw.get("writer", {})

    def _resolve_workspace(self) -> Path:
        ws = self._raw.get("workspace_dir", "./workspace")
        p = Path(ws)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        (p / "tasks").mkdir(exist_ok=True)
        return p

    def get_model(self, key: str | None = None) -> ModelConfig:
        """获取模型配置，缺省用 default_model。"""
        key = key or self.default_model
        if key not in self.model_registry:
            raise ValueError(
                f"模型 '{key}' 未在 config.yaml 注册。"
                f"可用: {list(self.model_registry.keys())}"
            )
        cfg = self.model_registry[key]
        if not cfg.available:
            raise ValueError(
                f"模型 '{key}' 的 API key 未配置（环境变量 {cfg.api_key_env}）。"
                f"请在 .env 中填写。"
            )
        return cfg

    def list_models(self) -> list[dict[str, Any]]:
        """列出所有模型及其可用状态，供前端展示。"""
        return [
            {
                "key": cfg.key,
                "provider": cfg.provider,
                "model": cfg.model,
                "available": cfg.available,
            }
            for cfg in self.model_registry.values()
        ]

    @classmethod
    def get(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_settings() -> Settings:
    return Settings.get()
