"""实时查看某任务的大模型消息与流式输出。

用法（在新开的命令行窗口里运行）:
  python scripts/watch_logs.py             # 默认跟最新任务，先回放已有日志再实时跟随
  python scripts/watch_logs.py <task_id>   # 指定任务
  python scripts/watch_logs.py --follow    # 只看新增（不回放历史）
  python scripts/watch_logs.py --replay    # 只回放历史（不跟随）

事件来源: workspace/tasks/<task_id>/logs/<agent>.jsonl
- type=messages : 发给大模型的 system+user 消息（prompt）
- type=delta    : 大模型流式返回的分片（逐块拼接 = 完整回答）
- 其它(start/done/plan/stage_done/exec_error/...) : 流程摘要
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = PROJECT_ROOT / "workspace"

# Windows 控制台默认 GBK，会乱码并在遇到特殊 Unicode(如公式符号)时崩溃 -> 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def latest_task() -> str | None:
    tasks = sorted(glob.glob(str(WORKSPACE / "tasks" / "*")))
    tasks = [t for t in tasks if os.path.isdir(t) and os.path.isdir(os.path.join(t, "logs"))]
    return tasks[-1] if tasks else None


def print_event(agent: str, ev: dict) -> None:
    t = ev.get("type", "?")
    ts = (ev.get("ts") or "")[11:19]
    if t == "messages":
        print(f"\n{'='*70}\n[{ts}] [{agent}] >>> 发给大模型的消息")
        for m in ev.get("messages", []):
            role = m.get("role", "")
            content = m.get("content", "")
            tag = {"system": "系统", "user": "用户", "assistant": "助手"}.get(role, role)
            print(f"--- ({tag}) ---\n{content}")
    elif t == "delta":
        sys.stdout.write(ev.get("text", ""))
        sys.stdout.flush()
    elif t == "done":
        extra = ev.get("extra", {})
        print(f"\n[{ts}] [{agent}] ✓ 完成: {ev.get('summary','')}")
        if extra:
            print(f"    extra: {json.dumps(extra, ensure_ascii=False)[:300]}")
    elif t == "error":
        print(f"\n[{ts}] [{agent}] ✗ 错误: {ev.get('error','')}")
    else:
        rest = {k: v for k, v in ev.items() if k not in ("type", "ts")}
        print(f"\n[{ts}] [{agent}] · {t}: {json.dumps(rest, ensure_ascii=False)[:400]}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    follow_only = "--follow" in sys.argv
    replay_only = "--replay" in sys.argv
    follow = not replay_only

    if args:
        task_dir = WORKSPACE / "tasks" / args[0]
    else:
        latest = latest_task()
        if not latest:
            print("未找到任何任务。先运行一次任务再来看。")
            return
        task_dir = Path(latest)

    logs_dir = task_dir / "logs"
    if not logs_dir.is_dir():
        print(f"任务目录无 logs: {task_dir}")
        return

    print(f"监听任务: {task_dir.name}\n{'='*70}")

    offsets: dict[str, int] = {}
    while True:
        for f in glob.glob(str(logs_dir / "*.jsonl")):
            if f not in offsets:
                # 新文件: --follow 从末尾开始(只看新增), 否则从头回放
                offsets[f] = os.path.getsize(f) if follow_only else 0
            agent = Path(f).stem
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    fh.seek(offsets[f])
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            print_event(agent, json.loads(line))
                        except json.JSONDecodeError:
                            continue
                    offsets[f] = fh.tell()
            except FileNotFoundError:
                pass
        if not follow:
            break
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n停止监听。")
