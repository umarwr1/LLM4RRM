import json

import numpy as np
import pytest

from llm4rrm.beamforming.parsing import parse_evaluator_feedback, parse_hyperparams, parse_solver_reply
from llm4rrm.beamforming.prompts import PromptBuilder
from llm4rrm.config import Config
from llm4rrm.core.parsing import ParseError

M, U, N = 4, 4, 8


def _bf_reply(shape=(M, U)):
    obj = {"wr": np.zeros(shape).tolist(), "wi": np.zeros(shape).tolist(), "alpha": 0.5}
    return "Reasoning about the problem.\n" + json.dumps(obj)


def test_solver_parse_ok_and_reasoning():
    out = parse_solver_reply(_bf_reply(), ["wr", "wi", "alpha"], M, U, N, 1000)
    assert out.block["wr"].shape == (M, U) and "Reasoning" in out.reasoning


def test_solver_parse_wrong_shape_rejected():
    with pytest.raises(ParseError):
        parse_solver_reply(_bf_reply((M, M + 4)), ["wr", "wi", "alpha"], M, U, N, 1000)


def test_solver_parse_ris_block():
    obj = {"a": [0.0] * N, "phi_t": [0.1] * N, "phi_r": [0.2] * N}
    out = parse_solver_reply("text " + json.dumps(obj), ["a", "phi_t", "phi_r"], M, U, N, 1000)
    assert out.block["phi_r"].shape == (N,)


def test_solver_parse_rejects_nonfinite():
    with pytest.raises(ParseError):
        parse_solver_reply('{"a": [NaN, 0, 0, 0, 0, 0, 0, 0], "phi_t": [0,0,0,0,0,0,0,0], "phi_r": [0,0,0,0,0,0,0,0]}', ["a", "phi_t", "phi_r"], M, U, N, 1000)


def test_evaluator_feedback_tagged_and_untagged():
    fb = parse_evaluator_feedback("<FEEDBACK_AGENT_1>raise alpha</FEEDBACK_AGENT_1>\n<FEEDBACK_AGENT_2>fix phases</FEEDBACK_AGENT_2>", 2)
    assert fb == ["raise alpha", "fix phases"]
    fb = parse_evaluator_feedback("some analysis. FEEDBACK: do X", 2)
    assert len(fb) == 2 and "do X" in fb[0]


def test_evaluator_feedback_line_limit():
    text = "<FEEDBACK_AGENT_1>" + "\n".join(f"l{i}" for i in range(10)) + "</FEEDBACK_AGENT_1>"
    assert len(parse_evaluator_feedback(text, 1, max_lines=5)[0].splitlines()) == 5


def test_hyperparams_parse():
    out = parse_hyperparams('I think we need more momentum.\n{"mu": 0.05, "gamma": 0.8}')
    assert (out.mu, out.gamma) == (0.05, 0.8)
    with pytest.raises(ParseError):
        parse_hyperparams("no json here")


@pytest.mark.parametrize("level", range(1, 8))
def test_prompt_levels_are_cumulative(level):
    cfg = Config().with_overrides({"framework.prompt_level": level})
    pb = PromptBuilder(cfg.system, cfg.framework, cfg.projection)
    text = pb.solver_system(1)
    assert isinstance(text, str) and len(text) > 100
    # feature flags follow Table 4
    assert cfg.framework.eff_n_fewshot == (0 if level < 2 else 3)
    assert cfg.framework.eff_use_evaluator == (level >= 7)
    assert cfg.framework.eff_num_solvers == (1 if level < 6 else 2)


def test_prompts_do_not_grow_shorter_with_level():
    lens = []
    for level in range(1, 6):
        cfg = Config().with_overrides({"framework.prompt_level": level})
        lens.append(len(PromptBuilder(cfg.system, cfg.framework, cfg.projection).solver_system(1)))
    assert lens == sorted(lens)
