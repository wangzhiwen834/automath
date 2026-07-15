"use client";
import Link from "next/link";
import { Network } from "lucide-react";

export default function NavBar({ active }: { active?: "home" | "task" }) {
  return (
    <nav className="sticky top-0 z-20 backdrop-blur bg-white/80 border-b border-[var(--border)]">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center gap-3">
        <Link href="/" className="flex items-center gap-2 group">
          <span className="w-8 h-8 rounded-lg brand-gradient flex items-center justify-center text-white">
            <Network size={18} />
          </span>
          <span className="font-bold text-slate-800 group-hover:text-indigo-600 transition">
            MathModeling&nbsp;Agent
          </span>
        </Link>
        <span className="text-xs text-slate-400 hidden sm:inline">多Agent协作数学建模自动化</span>
        <div className="flex-1" />
        <a href="http://localhost:8000/docs" target="_blank"
          className="text-xs text-slate-500 hover:text-indigo-600">API 文档</a>
      </div>
    </nav>
  );
}
