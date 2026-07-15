"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { api, type TaskState, type AgentName } from "./api";

// 每个 Agent 的流式文本累积
export type StreamMap = Record<string, string>;

export function useTaskEvents(taskId: string | null) {
  const [streams, setStreams] = useState<StreamMap>({});
  const [state, setState] = useState<TaskState | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  // 连接 WebSocket
  useEffect(() => {
    if (!taskId) return;
    setStreams({});
    setEvents([]);

    const ws = new WebSocket(api.wsUrl(taskId));
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (msg) => {
      const ev = JSON.parse(msg.data);
      setEvents((prev) => [...prev, ev]);

      if (ev.type === "snapshot" || ev.type === "final") {
        if (ev.state) setState(ev.state);
      }
      // 回放或实时 delta：累积到对应 agent
      if (ev.type === "replay") {
        if (ev.text) {
          setStreams((p) => ({ ...p, [ev.agent]: (p[ev.agent] || "") + ev.text }));
        }
      } else if (ev.type === "delta") {
        setStreams((p) => ({ ...p, [ev.agent]: (p[ev.agent] || "") + ev.text }));
      } else if (ev.type === "agent_done" || ev.type === "review" || ev.type === "completed" || ev.type === "paused" || ev.type === "failed") {
        // 状态变更后拉取最新 task detail
        fetch(`${api.wsUrl(taskId).replace("ws", "http")}/api/tasks/${taskId}`)
          .then((r) => r.json())
          .then((d) => setState(d.state))
          .catch(() => {});
      }
    };

    return () => ws.close();
  }, [taskId]);

  return { streams, state, connected, events };
}
