// 后端 API 客户端

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export type AgentName = "analyst" | "modeler" | "solver" | "summarizer" | "writer" | "reviewer";
export type TaskStatus = "created" | "running" | "paused" | "completed" | "failed" | "cancelled";
export type RunMode = "auto" | "interactive" | "hybrid";

export interface ModelInfo {
  key: string;
  provider: string;
  model: string;
  available: boolean;
}

export interface TaskSummary {
  task_id: string;
  title: string;
  mode: RunMode;
  problem_type: string;
  status: TaskStatus;
  current_agent: AgentName | null;
  created_at: string;
  updated_at?: string;
}

export interface AgentRecord {
  agent: AgentName;
  status: "pending" | "running" | "done" | "failed" | "skipped";
  model: string | null;
  artifact_path: string | null;
  summary: string | null;
  review_score: number | null;
  review_passed: boolean | null;
  error: string | null;
}

export interface TaskState {
  status: TaskStatus;
  current_agent: AgentName | null;
  agents: Record<string, AgentRecord>;
  review_round: number;
  waiting_for_human: boolean;
  human_decision: string | null;
}

export interface TaskDetail {
  meta: {
    task_id: string;
    title: string;
    mode: RunMode;
    problem_type: string;
    agent_models: Record<string, string>;
    created_at: string;
  };
  state: TaskState;
  running: boolean;
}

export const AGENT_LABELS: Record<AgentName, string> = {
  analyst: "问题分析",
  modeler: "建模",
  solver: "求解",
  summarizer: "总结",
  writer: "写作",
  reviewer: "审查",
};

export const AGENT_ORDER: AgentName[] = ["analyst", "modeler", "solver", "summarizer", "writer", "reviewer"];

export const MODE_LABELS: Record<RunMode, string> = {
  auto: "A 全自动",
  interactive: "B 人在回路",
  hybrid: "C 混合",
};

async function jget<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  listModels: () => jget<{ models: ModelInfo[]; default: string }>(`${API_BASE}/api/models`),
  listTasks: () => jget<{ tasks: TaskSummary[] }>(`${API_BASE}/api/tasks`),
  getTask: (id: string) => jget<TaskDetail>(`${API_BASE}/api/tasks/${id}`),
  createTask: (body: {
    title: string; problem_text: string; mode: RunMode; problem_type: string;
  }) => fetch(`${API_BASE}/api/tasks`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }).then((r) => r.json()),
  runTask: (id: string) =>
    fetch(`${API_BASE}/api/tasks/${id}/run`, { method: "POST" })
      .then(async (r) => ({ ok: r.ok, status: r.status, ...(await r.json()) })),
  resumeTask: (id: string, decision: string, feedback?: string) =>
    fetch(`${API_BASE}/api/tasks/${id}/resume`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, feedback }),
    }).then((r) => r.json()),
  deleteTask: (id: string) => fetch(`${API_BASE}/api/tasks/${id}`, { method: "DELETE" }).then((r) => r.json()),
  getArtifact: (id: string, agent: AgentName) =>
    fetch(`${API_BASE}/api/tasks/${id}/artifacts/${agent}`).then((r) => r.text()),
  getLogs: (id: string, agent: AgentName) =>
    jget<{ events: any[] }>(`${API_BASE}/api/tasks/${id}/logs/${agent}`),
  getFigures: (id: string) => jget<{ figures: string[] }>(`${API_BASE}/api/tasks/${id}/figures`),
  getDataFiles: (id: string) => jget<{ files: string[] }>(`${API_BASE}/api/tasks/${id}/data-files`),
  uploadFiles: (id: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    return fetch(`${API_BASE}/api/tasks/${id}/upload`, { method: "POST", body: fd }).then((r) => r.json());
  },
  figureUrl: (id: string, name: string) => `${API_BASE}/api/tasks/${id}/figures/${name}`,
  paperHtmlUrl: (id: string) => `${API_BASE}/api/tasks/${id}/paper.html`,
  paperPdfUrl: (id: string) => `${API_BASE}/api/tasks/${id}/paper.pdf`,
  wsUrl: (id: string) =>
    `${API_BASE.replace("http", "ws")}/ws/tasks/${id}`,
};
