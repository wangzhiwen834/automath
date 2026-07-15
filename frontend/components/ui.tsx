"use client";
import { CheckCircle2, Circle, Loader2, Pause, XCircle, Slash } from "lucide-react";
import type { TaskStatus } from "@/lib/api";

export const STATUS_META: Record<string, { label: string; cls: string; icon: any }> = {
  created: { label: "待启动", cls: "bg-slate-100 text-slate-600", icon: Circle },
  running: { label: "运行中", cls: "bg-blue-50 text-blue-600", icon: Loader2 },
  paused: { label: "已暂停", cls: "bg-amber-50 text-amber-600", icon: Pause },
  completed: { label: "已完成", cls: "bg-emerald-50 text-emerald-600", icon: CheckCircle2 },
  failed: { label: "失败", cls: "bg-rose-50 text-rose-600", icon: XCircle },
  cancelled: { label: "已取消", cls: "bg-slate-100 text-slate-400", icon: Slash },
};

export function StatusBadge({ status }: { status: string }) {
  const m = STATUS_META[status] || STATUS_META.created;
  const Icon = m.icon;
  const spin = status === "running";
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full ${m.cls}`}>
      <Icon size={12} className={spin ? "animate-spin" : ""} />
      {m.label}
    </span>
  );
}

export const TYPE_LABELS: Record<string, string> = {
  optimization: "优化类",
  evaluation: "评价类",
  statistics: "统计预测",
  mechanism: "机理建模",
  graph: "图论网络",
  unknown: "未分类",
};
