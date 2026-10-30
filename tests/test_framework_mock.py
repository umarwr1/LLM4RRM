import json

from llm4rrm.beamforming.runner import load_or_build_memory, run_experiment
from llm4rrm.config import Config
from llm4rrm.llm.factory import create_llm_client


def _cfg(tmp_path, **extra):
    ov = {"llm.provider": "mock", "experiment.n_instances": 2, "experiment.memory_bank_size": 6,
          "experiment.memory_bank_path": str(tmp_path / "bank.npz"), "experiment.output_dir": str(tmp_path / "out"),
          "system.M": 4, "system.N": 8, "system.U": 4, "system.r_min_mbps": 1.0}
    ov.update(extra)
    return Config().with_overrides(ov)


def test_end_to_end_mock_run(tmp_path):
    cfg = _cfg(tmp_path)
    client = create_llm_client(cfg.llm)
    summary = run_experiment(cfg, client, load_or_build_memory(cfg, verbose=False), verbose=False)
    agg = summary["aggregate"]
    assert agg["n"] == 2 and agg["n_failed"] == 0
    assert agg["V_post_pct"] == 0.0  # projection guarantees feasibility on these easy instances
    rows = [json.loads(l) for l in (tmp_path / "out" / "instances.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["n_agent_failures"] == 0
    assert rows[0]["llm"]["calls"] <= 10 + 4 + 5 + 2  # K*T solver + <=T-1 evaluator + <=L hyper (+slack)


def test_call_budget_per_configuration(tmp_path):
    for k, ev in ((1, False), (2, False), (2, True)):
        cfg = _cfg(tmp_path, **{"framework.num_solvers": k, "framework.use_evaluator": ev, "experiment.output_dir": str(tmp_path / f"o{k}{ev}")})
        client = create_llm_client(cfg.llm)
        run_experiment(cfg, client, load_or_build_memory(cfg, verbose=False), verbose=False)
        row = json.loads((tmp_path / f"o{k}{ev}" / "instances.jsonl").read_text().splitlines()[0])
        solver_calls = k * cfg.framework.T
        assert row["llm"]["calls"] >= solver_calls
        if not ev:
            assert row["llm"]["calls"] <= solver_calls + cfg.projection.L


def test_prompt_level_1_needs_no_memory(tmp_path):
    cfg = _cfg(tmp_path, **{"framework.prompt_level": 1})
    assert load_or_build_memory(cfg, verbose=False) is None
