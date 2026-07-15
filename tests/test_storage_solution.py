from app.storage import AgentName


def test_subproblem_dir_created(make_store_task):
    store, task = make_store_task("题目")
    d = store.subproblem_dir(task.meta.task_id, "sub1")
    assert d.exists() and d.is_dir()
    assert d.name == "sub1"
    assert d.parent.name == "solution"


def test_write_and_read_solution_file(make_store_task):
    store, task = make_store_task("题目")
    path = store.write_solution_file(task.meta.task_id, "plan.json", '{"a":1}')
    assert path == "artifacts/solution/plan.json"
    assert store.read_solution_file(task.meta.task_id, "plan.json") == '{"a":1}'


def test_write_artifact_solver_writes_manifest(make_store_task):
    store, task = make_store_task("题目")
    path = store.write_artifact(task.meta.task_id, AgentName.SOLVER, '{"subs":[]}')
    assert path == "artifacts/solution/manifest.json"
    assert store.read_artifact(task.meta.task_id, path) == '{"subs":[]}'
