"""验证脚本：检查多模型配置与连通性。

用法:
    python scripts/verify_models.py            # 列出所有模型状态
    python scripts/verify_models.py deepseek-chat  # 测试指定模型对话

不传 key 时只列状态不发请求，不消耗 token。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能 import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 控制台默认 GBK，强制 UTF-8 避免中文/符号打印报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.config import get_settings
from app.llm import get_llm
from app.llm.provider import Message


def list_models() -> None:
    settings = get_settings()
    print(f"\n工作目录: {settings.workspace_dir}")
    print(f"默认模型: {settings.default_model}\n")
    print(f"{'模型key':<20} {'provider':<18} {'模型名':<22} {'可用'}")
    print("-" * 70)
    for m in settings.list_models():
        flag = "✓ 已配置" if m["available"] else "✗ 缺API key"
        print(f"{m['key']:<20} {m['provider']:<18} {m['model']:<22} {flag}")
    print()


def test_chat(model_key: str) -> None:
    print(f"\n测试模型 [{model_key}] ...")
    llm = get_llm(model_key)
    messages = [
        Message("system", "你是一名数学建模专家，回答简洁。"),
        Message("user", "用一句话说明什么是层次分析法(AHP)。"),
    ]
    print("\n--- 流式输出 ---")
    full = ""
    for chunk in llm.stream(messages, max_tokens=200):
        print(chunk, end="", flush=True)
        full += chunk
    print("\n--- 流式结束 ---\n")
    print(f"输出长度: {len(full)} 字符\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_chat(sys.argv[1])
    else:
        list_models()
        print("提示: 运行 `python scripts/verify_models.py <模型key>` 测试对话连通性")
