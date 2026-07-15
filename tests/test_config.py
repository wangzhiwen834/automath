from app.config import get_settings


def test_solver_and_writer_config_loaded():
    s = get_settings()
    assert s.solver_config.get("max_stage_retries") == 2
    assert s.solver_config.get("stage_execution_timeout") == 120
    assert s.solver_config.get("preamble_seed") == 42
    assert s.writer_config.get("max_expand_sections") == 4
    assert s.writer_config.get("consistency_check") is True
