import TaskDetail from "@/components/TaskDetail";

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <TaskDetail taskId={id} />;
}
