"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Plus, ListChecks, Sparkles, FileUp, Rocket, Clock, ChevronRight, Trash2,
} from "lucide-react";
import NavBar from "@/components/NavBar";
import { StatusBadge, TYPE_LABELS } from "@/components/ui";
import { api, type TaskSummary, type RunMode } from "@/lib/api";

const MODES: { key: RunMode; label: string; desc: string }[] = [
  { key: "auto", label: "A · 全自动", desc: "一路跑到底" },
  { key: "hybrid", label: "C · 混合", desc: "关键节点确认" },
  { key: "interactive", label: "B · 人在回路", desc: "每步确认" },
];

const TYPES = [
  { key: "optimization", label: "优化类" },
  { key: "evaluation", label: "评价类" },
  { key: "statistics", label: "统计预测" },
  { key: "mechanism", label: "机理建模" },
  { key: "graph", label: "图论网络" },
  { key: "unknown", label: "未分类" },
];

export default function Home() {
  const router = useRouter();
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const [title, setTitle] = useState("");
  const [problem, setProblem] = useState("");
  const [mode, setMode] = useState<RunMode>("auto");
  const [ptype, setPtype] = useState("mechanism");
  const [files, setFiles] = useState<File[]>([]);
  const [creating, setCreating] = useState(false);

  const load = () => api.listTasks().then((d) => setTasks(d.tasks)).finally(() => setLoading(false));
  useEffect(() => { load(); const t = setInterval(load, 3000); return () => clearInterval(t); }, []);

  const stats = {
    total: tasks.length,
    running: tasks.filter((t) => t.status === "running").length,
    done: tasks.filter((t) => t.status === "completed").length,
  };

  const submit = async () => {
    if (!title || !problem) { alert("请填写标题和题目"); return; }
    setCreating(true);
    try {
      const r = await api.createTask({ title, problem_text: problem, mode, problem_type: ptype });
      if (files.length) await api.uploadFiles(r.task_id, files);
      router.push(`/tasks/${r.task_id}`);
    } catch (e) {
      alert("创建失败: " + e);
    } finally {
      setCreating(false);
    }
  };

  const del = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("删除该任务？")) return;
    await api.deleteTask(id);
    load();
  };

  return (
    <>
      <NavBar active="home" />
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-6">
        {/* 头部统计 */}
        <div className="card brand-gradient text-white p-6 mb-6 relative overflow-hidden">
          <div className="relative z-10">
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Sparkles size={22} /> 数学建模自动化工作台
            </h1>
            <p className="text-indigo-100 text-sm mt-1">
              分析 · 建模 · 求解 · 写作 · 审查 —— 五Agent协作，自动产出论文与图表
            </p>
            <div className="flex gap-6 mt-4 text-sm">
              <div><div className="text-2xl font-bold">{stats.total}</div><div className="text-indigo-200 text-xs">总任务</div></div>
              <div><div className="text-2xl font-bold">{stats.running}</div><div className="text-indigo-200 text-xs">运行中</div></div>
              <div><div className="text-2xl font-bold">{stats.done}</div><div className="text-indigo-200 text-xs">已完成</div></div>
            </div>
          </div>
          <div className="absolute -right-8 -bottom-10 opacity-20">
            <Plus size={160} />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* 新建任务 */}
          <section className="card p-6 lg:col-span-3">
            <h2 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
              <Plus size={18} className="text-indigo-600" /> 新建建模任务
            </h2>
            <div className="space-y-4 text-sm">
              <div>
                <label className="block text-slate-500 mb-1.5 text-xs font-medium">任务标题</label>
                <input className="w-full border border-[var(--border)] rounded-lg px-3 py-2.5 focus:ring-2 focus:ring-indigo-200 focus:border-indigo-400 outline-none transition"
                  placeholder="例如：2025高教社杯A题 烟幕干扰弹投放策略"
                  value={title} onChange={(e) => setTitle(e.target.value)} />
              </div>
              <div>
                <label className="block text-slate-500 mb-1.5 text-xs font-medium">题目内容</label>
                <textarea className="w-full border border-[var(--border)] rounded-lg px-3 py-2.5 h-36 font-mono text-xs focus:ring-2 focus:ring-indigo-200 focus:border-indigo-400 outline-none transition"
                  placeholder="粘贴题目全文（含问题1、问题2…）" value={problem}
                  onChange={(e) => setProblem(e.target.value)} />
              </div>

              {/* 模式选择 */}
              <div>
                <label className="block text-slate-500 mb-1.5 text-xs font-medium">运行模式</label>
                <div className="grid grid-cols-3 gap-2">
                  {MODES.map((m) => (
                    <button key={m.key} onClick={() => setMode(m.key)}
                      className={`text-left px-3 py-2 rounded-lg border transition ${mode === m.key
                        ? "border-indigo-500 bg-indigo-50 ring-2 ring-indigo-100"
                        : "border-[var(--border)] hover:border-indigo-300"}`}>
                      <div className="font-medium text-slate-800 text-sm">{m.label}</div>
                      <div className="text-xs text-slate-400">{m.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* 题型 */}
              <div>
                <label className="block text-slate-500 mb-1.5 text-xs font-medium">题型</label>
                <div className="flex flex-wrap gap-2">
                  {TYPES.map((t) => (
                    <button key={t.key} onClick={() => setPtype(t.key)}
                      className={`px-3 py-1.5 rounded-full text-xs border transition ${ptype === t.key
                        ? "border-indigo-500 bg-indigo-500 text-white"
                        : "border-[var(--border)] text-slate-600 hover:border-indigo-300"}`}>
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* 数据集上传 */}
              <div>
                <label className="block text-slate-500 mb-1.5 text-xs font-medium">数据集（可选）</label>
                <label className="flex items-center gap-3 border border-dashed border-[var(--border)] rounded-lg px-4 py-3 cursor-pointer hover:border-indigo-400 hover:bg-indigo-50/40 transition">
                  <FileUp size={18} className="text-indigo-500" />
                  <span className="text-sm text-slate-500">
                    {files.length ? files.map((f) => f.name).join(", ") : "点击上传 csv / xlsx / txt（可多选）"}
                  </span>
                  <input type="file" multiple className="hidden"
                    onChange={(e) => setFiles(Array.from(e.target.files || []))} />
                </label>
              </div>

              <button disabled={creating} onClick={submit}
                className="w-full brand-gradient text-white rounded-lg py-2.5 font-medium flex items-center justify-center gap-2 hover:opacity-90 disabled:opacity-50 transition">
                <Rocket size={16} /> {creating ? "创建中…" : "创建并启动"}
              </button>
            </div>
          </section>

          {/* 任务列表 */}
          <section className="card p-6 lg:col-span-2">
            <h2 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
              <ListChecks size={18} className="text-indigo-600" /> 历史任务
            </h2>
            {loading ? (
              <p className="text-slate-400 text-sm">加载中…</p>
            ) : tasks.length === 0 ? (
              <div className="text-center py-12 text-slate-400 text-sm">
                <Plus size={32} className="mx-auto mb-2 opacity-40" />
                暂无任务，左侧创建一个吧
              </div>
            ) : (
              <ul className="space-y-2 max-h-[560px] overflow-auto pr-1">
                {tasks.map((t) => (
                  <li key={t.task_id}>
                    <div onClick={() => router.push(`/tasks/${t.task_id}`)}
                      className="group cursor-pointer border border-[var(--border)] rounded-lg px-3 py-2.5 hover:border-indigo-300 hover:shadow-sm transition">
                      <div className="flex justify-between items-start gap-2">
                        <span className="font-medium text-sm text-slate-800 truncate flex-1">{t.title}</span>
                        <StatusBadge status={t.status} />
                      </div>
                      <div className="flex items-center gap-2 mt-1.5 text-xs text-slate-400 flex-wrap">
                        <span className="px-1.5 py-0.5 bg-slate-100 rounded">{TYPE_LABELS[t.problem_type] || t.problem_type}</span>
                        <span className="px-1.5 py-0.5 bg-slate-100 rounded">{t.mode}</span>
                        <span className="flex items-center gap-1"><Clock size={11} />{t.created_at.replace("T", " ").slice(5, 16)}</span>
                        <button onClick={(e) => del(t.task_id, e)}
                          className="ml-auto text-slate-300 hover:text-rose-500 opacity-0 group-hover:opacity-100 transition">
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </main>
    </>
  );
}
