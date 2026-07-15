"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft, Search, FunctionSquare, Terminal, FileText, ShieldCheck,
  Download, Eye, Play, CheckCircle2, AlertTriangle, FileUp, Image as ImageIcon,
  RotateCcw, Cpu, History,
} from "lucide-react";
import NavBar from "@/components/NavBar";
import { StatusBadge } from "@/components/ui";
import MarkdownView from "./MarkdownView";
import { api, AGENT_ORDER, AGENT_LABELS, type TaskDetail as TaskDetailT, type AgentName } from "@/lib/api";
import { useTaskEvents } from "@/lib/useTaskEvents";

const AGENT_ICONS: Record<AgentName, any> = {
  analyst: Search, modeler: FunctionSquare, solver: Terminal, writer: FileText, reviewer: ShieldCheck,
};

const AGENT_STATUS_STYLE: Record<string, string> = {
  pending: "text-slate-300 border-slate-200 bg-white",
  running: "text-indigo-600 border-indigo-400 bg-indigo-50",
  done: "text-emerald-600 border-emerald-400 bg-emerald-50",
  failed: "text-rose-600 border-rose-400 bg-rose-50",
  skipped: "text-slate-300 border-slate-200 bg-slate-50",
};

export default function TaskDetail({ taskId }: { taskId: string }) {
  const [detail, setDetail] = useState<TaskDetailT | null>(null);
  const [active, setActive] = useState<AgentName>("analyst");
  const [artifacts, setArtifacts] = useState<Record<string, string>>({});
  const [figures, setFigures] = useState<string[]>([]);
  const [dataFiles, setDataFiles] = useState<string[]>([]);
  const [feedback, setFeedback] = useState("");
  const [busy, setBusy] = useState(false);

  const { streams, state, connected } = useTaskEvents(taskId);

  const reload = () => {
    api.getTask(taskId).then((d) => setDetail(d)).catch(() => {});
    api.getFigures(taskId).then((d) => setFigures(d.figures)).catch(() => {});
    api.getDataFiles(taskId).then((d) => setDataFiles(d.files)).catch(() => {});
  };
  useEffect(() => { reload(); const t = setInterval(reload, 4000); return () => clearInterval(t); }, [taskId]);

  useEffect(() => {
    if (!detail) return;
    AGENT_ORDER.forEach((a) => {
      const rec = state?.agents[a] || detail.state.agents[a];
      if (rec?.status === "done" && rec.artifact_path && artifacts[a] === undefined) {
        api.getArtifact(taskId, a).then((txt) => setArtifacts((p) => ({ ...p, [a]: txt })));
      }
    });
  }, [state, detail]);

  const mergedState = state || detail?.state;
  const status = mergedState?.status || detail?.state.status;
  const waiting = mergedState?.waiting_for_human;
  const reviewRound = mergedState?.review_round || 0;

  const run = async () => { setBusy(true); await api.runTask(taskId); setBusy(false); reload(); };
  const resume = async (decision: string) => {
    setBusy(true);
    await api.resumeTask(taskId, decision, decision === "modify" ? feedback : undefined);
    setFeedback(""); setBusy(false); reload();
  };
  const onUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fs = Array.from(e.target.files || []);
    if (fs.length) { await api.uploadFiles(taskId, fs); reload(); }
  };

  const activeRec = mergedState?.agents[active];
  const activeStream = streams[active];
  const activeContent = artifacts[active] ?? (activeRec?.status === "running" ? activeStream : "") ?? "";
  const isStreaming = activeRec?.status === "running" && !!activeStream;

  // 进度
  const doneCount = AGENT_ORDER.filter((a) => mergedState?.agents[a]?.status === "done").length;
  const progress = Math.round((doneCount / AGENT_ORDER.length) * 100);

  return (
    <>
      <NavBar />
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-5">
        {/* 头部信息卡 */}
        <div className="card p-5 mb-4">
          <div className="flex items-center gap-3 flex-wrap">
            <Link href="/" className="text-slate-400 hover:text-indigo-600"><ArrowLeft size={18} /></Link>
            <h1 className="font-bold text-slate-800 text-lg flex-1 min-w-0 truncate">
              {detail?.meta.title || "加载中…"}
            </h1>
            <StatusBadge status={status || "created"} />
            <span className="text-xs text-slate-400 flex items-center gap-1">
              <Cpu size={12} /> {detail?.meta.mode}
            </span>
            {status === "created" && (
              <button onClick={run} disabled={busy}
                className="brand-gradient text-white text-sm px-4 py-1.5 rounded-lg flex items-center gap-1.5 hover:opacity-90 disabled:opacity-50">
                <Play size={14} /> 启动
              </button>
            )}
            {status === "completed" && (
              <a href={api.paperPdfUrl(taskId)} target="_blank"
                className="bg-emerald-600 text-white text-sm px-3 py-1.5 rounded-lg flex items-center gap-1.5 hover:bg-emerald-700">
                <Download size={14} /> 下载PDF
              </a>
            )}
            <a href={api.paperHtmlUrl(taskId)} target="_blank"
              className="border border-[var(--border)] text-sm px-3 py-1.5 rounded-lg flex items-center gap-1.5 hover:bg-slate-50 text-slate-600">
              <Eye size={14} /> 预览
            </a>
          </div>
          {/* 进度条 */}
          <div className="mt-4">
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>流水线进度</span>
              <span>{doneCount}/{AGENT_ORDER.length} · {progress}%{reviewRound > 0 && ` · 回退${reviewRound}轮`}</span>
            </div>
            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
              <div className="h-full brand-gradient rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
            </div>
          </div>
        </div>

        {/* 流水线 stepper */}
        <div className="card p-4 mb-4">
          <div className="flex items-center gap-1 overflow-x-auto">
            {AGENT_ORDER.map((a, i) => {
              const rec = mergedState?.agents[a];
              const st = rec?.status || "pending";
              const Icon = AGENT_ICONS[a];
              const isActive = active === a;
              const isCurrent = mergedState?.current_agent === a && st === "running";
              return (
                <div key={a} className="flex items-center">
                  <button onClick={() => setActive(a)}
                    className={`flex flex-col items-center px-3 py-2 rounded-xl border-2 transition min-w-[88px] ${AGENT_STATUS_STYLE[st]} ${isActive ? "ring-2 ring-offset-1 ring-indigo-300" : ""}`}>
                    <Icon size={20} className={isCurrent ? "pulse-dot" : ""} />
                    <span className="text-xs font-medium mt-1 text-slate-700">{AGENT_LABELS[a]}</span>
                    <span className="text-[10px] text-slate-400">
                      {a === "reviewer" && rec?.review_score != null
                        ? `${rec.review_score}分${rec.review_passed ? "✓" : "✗"}`
                        : st === "done" ? "完成" : st === "running" ? "运行中" : st === "failed" ? "失败" : "待运行"}
                    </span>
                  </button>
                  {i < AGENT_ORDER.length - 1 && (
                    <div className={`h-0.5 w-6 ${mergedState?.agents[AGENT_ORDER[i + 1]]?.status === "done" || rec?.status === "done" ? "bg-emerald-400" : "bg-slate-200"}`} />
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* 人在回路决策栏 */}
        {waiting && (
          <div className="card border-amber-300 bg-amber-50/60 p-4 mb-4">
            <div className="flex items-center gap-2 font-semibold text-amber-800 mb-2">
              <AlertTriangle size={16} /> 已暂停，等待你的决策
            </div>
            <textarea className="w-full border border-amber-300 rounded-lg p-2.5 text-sm h-20 bg-white focus:ring-2 focus:ring-amber-200 outline-none"
              placeholder="如需修改，填写反馈意见（选填）" value={feedback}
              onChange={(e) => setFeedback(e.target.value)} />
            <div className="flex gap-2 mt-2">
              <button onClick={() => resume("approve")} disabled={busy}
                className="bg-emerald-600 text-white text-sm px-4 py-1.5 rounded-lg flex items-center gap-1.5 hover:bg-emerald-700 disabled:opacity-50">
                <CheckCircle2 size={14} /> 通过并继续
              </button>
              <button onClick={() => resume("modify")} disabled={busy}
                className="bg-amber-600 text-white text-sm px-4 py-1.5 rounded-lg flex items-center gap-1.5 hover:bg-amber-700 disabled:opacity-50">
                <RotateCcw size={14} /> 带反馈重做
              </button>
            </div>
          </div>
        )}

        {/* 主体 */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-4">
          {/* 内容区 */}
          <section className="card p-5 min-h-[460px]">
            <div className="flex justify-between items-center mb-3 pb-3 border-b border-[var(--border)]">
              <h2 className="font-semibold text-slate-800 flex items-center gap-2">
                {(() => { const I = AGENT_ICONS[active]; return <I size={16} className="text-indigo-600" />; })()}
                {AGENT_LABELS[active]}
                {activeRec?.model && <span className="text-xs font-normal text-slate-400 ml-1">· {activeRec.model}</span>}
              </h2>
              {isStreaming && (
                <span className="text-xs text-indigo-600 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full pulse-dot" /> 流式输出中
                </span>
              )}
            </div>
            <div className="text-sm overflow-auto max-h-[68vh]">
              {active === "solver" ? (
                <pre className={`text-xs font-mono whitespace-pre-wrap bg-slate-900 text-slate-100 p-4 rounded-lg ${isStreaming ? "stream-cursor" : ""}`}>{activeContent || "(运行中…)"}</pre>
              ) : (
                <div className={isStreaming ? "stream-cursor" : ""}><MarkdownView content={activeContent} /></div>
              )}
            </div>
          </section>

          {/* 侧栏 */}
          <aside className="space-y-4">
            {figures.length > 0 && (
              <section className="card p-4">
                <h3 className="font-semibold text-sm text-slate-700 mb-3 flex items-center gap-1.5">
                  <ImageIcon size={14} className="text-indigo-600" /> 生成的图表 ({figures.length})
                </h3>
                <div className="space-y-3">
                  {figures.map((f) => (
                    <a key={f} href={api.figureUrl(taskId, f)} target="_blank" className="block group">
                      <img src={api.figureUrl(taskId, f)} alt={f}
                        className="w-full rounded-lg border border-[var(--border)] group-hover:shadow-md transition" />
                      <p className="text-xs text-slate-400 text-center mt-1 truncate">{f}</p>
                    </a>
                  ))}
                </div>
              </section>
            )}

            <section className="card p-4">
              <h3 className="font-semibold text-sm text-slate-700 mb-2 flex items-center gap-1.5">
                <FileUp size={14} className="text-indigo-600" /> 数据文件
              </h3>
              {dataFiles.length === 0 ? (
                <p className="text-xs text-slate-400">未上传</p>
              ) : (
                <ul className="text-xs space-y-1 text-slate-600">
                  {dataFiles.map((f) => <li key={f} className="truncate">📄 {f}</li>)}
                </ul>
              )}
              <label className="mt-2 block text-center text-xs text-indigo-600 cursor-pointer border border-dashed border-indigo-200 rounded-lg py-1.5 hover:bg-indigo-50">
                + 追加上传
                <input type="file" multiple className="hidden" onChange={onUpload} />
              </label>
            </section>

            {reviewRound > 0 && (
              <section className="card p-4">
                <h3 className="font-semibold text-sm text-slate-700 mb-1 flex items-center gap-1.5">
                  <History size={14} className="text-indigo-600" /> 审查回退
                </h3>
                <p className="text-xs text-slate-500">已回退重做 {reviewRound} 轮</p>
              </section>
            )}
          </aside>
        </div>
      </main>
    </>
  );
}
