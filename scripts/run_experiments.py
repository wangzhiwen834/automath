"""批量消融实验运行器：在问题集 × 实验矩阵上跑完整流水线，收集指标到 JSON/CSV。

用法:
  python scripts/run_experiments.py                       # 跑全部 baseline+E1-E4
  python scripts/run_experiments.py --problems experiments/problems
  python scripts/run_experiments.py --only baseline,E1_monolithic_solver
  python scripts/run_experiments.py --models deepseek-v4-pro,qwen-plus  # E5 跨模型
  python scripts/run_experiments.py --out experiments/results

每次运行 = 1 道题 × 1 个配置 = 一条完整流水线（约 10-20 次 LLM 调用，消耗 token）。
配置覆盖通过直接改 get_settings() 单例实现（顺序执行，互不干扰）。

指标：求解成功率/阶段成功率/代码LOC/图数/论文字数/审查分/通过率/编造率(论文数字不在数据源中的比例)。
编造率用 write_utils.cross_check_numbers（与系统自检同源），有 50 vs 50.0、日期类 token 的已知假阳性，仅供横向对比。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.config import get_settings
from app.agents.write_utils import extract_numbers
from app.storage import RunMode, get_store
from app.orchestrator import Orchestrator

# ====================================================================
# 实验矩阵：name -> 配置覆盖（section.name -> 值）
# ====================================================================
EXPERIMENTS = [
    ("baseline", {}),
    ("E1_monolithic_solver", {"solver.architecture": "monolithic"}),
    ("E2_no_summarizer", {"pipeline.skip_summarizer": True}),
    ("E3_no_verification", {"solver.max_regen_per_stage": 1,
                            "solver.self_critique_enabled": False}),
    ("E4_no_consistency", {"writer.consistency_check": False}),
]


def apply_overrides(overrides: dict) -> None:
    """把 section.name 覆盖应用到 get_settings() 单例。"""
    s = get_settings()
    for key, val in overrides.items():
        section, _, name = key.partition(".")
        if section == "solver":
            s.solver_config[name] = val
        elif section == "writer":
            s.writer_config[name] = val
        elif section == "pipeline":
            s.pipeline_config[name] = val
        elif section == "reviewer":
            s.reviewer_config[name] = val
        else:
            raise ValueError(f"未知配置段: {section}")


def collect_metrics(task_id: str, exp_name: str, problem_name: str,
                    model_key: str, elapsed: float) -> dict:
    store = get_store()
    task = store.load(task_id)
    state = task.state

    rev = state.agents.get("reviewer", {})
    score = getattr(rev, "review_score", None)
    passed = getattr(rev, "review_passed", None)

    # 求解状态
    executed = None
    n_sub = None
    stages_ok = None
    stages_total = None
    status_raw = store.read_solution_file(task_id, "status.json")
    if status_raw:
        try:
            st = json.loads(status_raw)
            executed = st.get("executed")
            subs = st.get("subproblems", [])
            n_sub = len(subs)
            all_stages = [g for s in subs for g in s.get("stages", [])]
            stages_ok = sum(1 for g in all_stages if g.get("ok"))
            stages_total = len(all_stages)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    figures = store.list_figures(task_id)
    paper = store.read_artifact(task_id, "artifacts/paper.md")

    # 代码 LOC（solution 下所有 .py）
    loc = 0
    for py in store.solution_dir(task_id).rglob("*.py"):
        try:
            loc += len(py.read_text(encoding="utf-8").splitlines())
        except Exception:
            pass

    # 编造率：论文数字不在数据源(facts 优先, 否则 output)中的比例
    facts = store.read_artifact(task_id, "artifacts/facts.md")
    source = facts or store.read_artifact(task_id, "artifacts/solution/output.txt")
    paper_nums = extract_numbers(paper)
    src_nums = extract_numbers(source)
    fabricated = [n for n in paper_nums if n not in src_nums]
    fab_rate = len(fabricated) / len(paper_nums) if paper_nums else 0.0

    # 最终状态
    final_status = state.status.value if state.status else None

    return {
        "exp": exp_name,
        "problem": problem_name,
        "model": model_key,
        "task_id": task_id,
        "final_status": final_status,
        "executed": executed,
        "n_subproblems": n_sub,
        "stages_ok": stages_ok,
        "stages_total": stages_total,
        "n_figures": len(figures),
        "code_loc": loc,
        "paper_chars": len(paper),
        "review_score": score,
        "review_passed": passed,
        "paper_numbers": len(paper_nums),
        "fabricated_numbers": len(fabricated),
        "fabrication_rate": round(fab_rate, 4),
        "elapsed_sec": round(elapsed, 1),
    }


def run_one(problem_path: Path, exp_name: str, overrides: dict,
            model_key: str | None) -> dict:
    apply_overrides(overrides)
    s = get_settings()
    # E5 跨模型：覆盖所有 Agent 的模型
    agent_models = None
    if model_key:
        agent_models = {a: model_key for a in ["analyst", "modeler", "solver",
                                               "summarizer", "writer", "reviewer"]}

    problem_text = problem_path.read_text(encoding="utf-8")
    store = get_store()
    tag = exp_name if not model_key else f"{exp_name}__{model_key}"
    task = store.create_task(
        title=f"{problem_path.stem}__{tag}",
        problem_text=problem_text, mode=RunMode.AUTO,
        agent_models=agent_models,
    )
    print(f"\n{'='*70}\n▶ {problem_path.stem} | {tag} | task={task.meta.task_id}")

    def on_event(ev: dict) -> None:
        t = ev.get("type")
        if t == "agent_start":
            print(f"  ▶ [{ev['agent']}] 开始")
        elif t == "agent_done":
            print(f"  ✔ [{ev['agent']}] {ev.get('summary','')[:60]}")
        elif t == "agent_skipped":
            print(f"  ⏭ [{ev['agent']}] 跳过(消融)")
        elif t == "review":
            print(f"  🔍 审查 评分={ev.get('score')} 通过={ev.get('passed')} 回退={ev.get('rollback_to')}")
        elif t == "completed":
            print(f"  🏁 完成 评分={ev.get('score')} {ev.get('note','') or ''}")
        elif t == "failed":
            print(f"  ✗ 失败 {ev.get('agent')}: {str(ev.get('error',''))[:80]}")

    t0 = time.time()
    try:
        orch = Orchestrator(task.meta.task_id, event_sink=on_event)
        orch.run()
    except Exception as e:
        print(f"  ✗ 运行异常: {e}")
    elapsed = time.time() - t0

    metrics = collect_metrics(task.meta.task_id, exp_name, problem_path.stem,
                              model_key or s.default_model, elapsed)
    print(f"  指标: executed={metrics['executed']} stages={metrics['stages_ok']}/{metrics['stages_total']} "
          f"figs={metrics['n_figures']} loc={metrics['code_loc']} "
          f"paper={metrics['paper_chars']}字 评分={metrics['review_score']} "
          f"编造率={metrics['fabrication_rate']} 用时={metrics['elapsed_sec']}s")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="experiments/problems",
                    help="题目目录(每个 .txt 一道题)")
    ap.add_argument("--only", default="", help="只跑指定实验(逗号分隔)")
    ap.add_argument("--models", default="", help="E5 跨模型(逗号分隔 model key)；留空则不跑跨模型")
    ap.add_argument("--out", default="experiments/results", help="结果输出目录")
    args = ap.parse_args()

    pdir = Path(args.problems)
    problems = sorted(pdir.glob("*.txt"))
    if not problems:
        print(f"未在 {pdir} 找到题目(.txt)。请放入题目文件后重试。")
        sys.exit(1)

    exps = EXPERIMENTS
    if args.only:
        want = set(args.only.split(","))
        exps = [e for e in EXPERIMENTS if e[0] in want]
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else []

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    # baseline + E1-E4 用默认模型
    for exp_name, overrides in exps:
        for prob in problems:
            all_metrics.append(run_one(prob, exp_name, overrides, None))
            _flush(out_dir, all_metrics)
    # E5 跨模型：只跑 baseline 配置 × 各模型
    if models:
        for m in models:
            for prob in problems:
                all_metrics.append(run_one(prob, "E5_model", {}, m))
                _flush(out_dir, all_metrics)

    _flush(out_dir, all_metrics)
    _print_summary(all_metrics)
    print(f"\n结果已写入: {out_dir}/results.json 与 results.csv")


def _flush(out_dir: Path, metrics: list[dict]) -> None:
    if not metrics:
        return
    (out_dir / "results.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(metrics[0].keys())
    with open(out_dir / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in metrics:
            w.writerow(m)


def _print_summary(metrics: list[dict]) -> None:
    print(f"\n{'='*70}\n汇总（按实验聚合，各指标取均值）\n{'='*70}")
    by_exp: dict[str, list[dict]] = {}
    for m in metrics:
        by_exp.setdefault(m["exp"], []).append(m)
    hdr = f"{'exp':<22}{'n':>3}{'exec%':>7}{'stg_ok%':>8}{'figs':>6}{'loc':>7}{'paper':>8}{'score':>7}{'pass%':>7}{'fab%':>7}"
    print(hdr)
    for exp, ms in by_exp.items():
        n = len(ms)
        def avg(key):
            vals = [m[key] for m in ms if m[key] is not None]
            return sum(vals) / len(vals) if vals else 0
        exec_pct = avg("executed") * 100
        stg = avg("stages_ok") / avg("stages_total") * 100 if avg("stages_total") else 0
        pass_pct = avg("review_passed") * 100
        print(f"{exp:<22}{n:>3}{exec_pct:>6.0f}%{stg:>7.0f}%{avg('n_figures'):>6.0f}"
              f"{avg('code_loc'):>7.0f}{avg('paper_chars'):>8.0f}{avg('review_score'):>7.0f}"
              f"{pass_pct:>6.0f}%{avg('fabrication_rate')*100:>6.1f}%")


if __name__ == "__main__":
    main()
