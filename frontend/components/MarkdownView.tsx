"use client";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";

export default function MarkdownView({ content }: { content: string }) {
  if (!content) return <p className="text-slate-400 italic text-sm">（暂无内容）</p>;
  return (
    <div className="prose-paper">
      <ReactMarkdown remarkPlugins={[remarkMath, remarkGfm]} rehypePlugins={[rehypeKatex]}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
