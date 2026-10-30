"""LLM4RRM for joint beamforming and STAR-RIS phase-shift optimisation (paper Algorithm 1, Sec. III).

Pipeline per channel instance x = {G, H, sigma^2}:
  Step 1  retrieve few-shot sets S_M, S_E from the memory bank D (Frobenius-norm similarity);
  Step 2  K solver LLMs (K=2: beamforming agent then STAR-RIS agent) produce y_hat_k^(t) and reasoning traces;
  Step 3  evaluation block: objective F and constraint flags f_j for the joint solution;
  Step 4  evaluator LLM writes diagnostic feedback Delta_k^(t+1) for every solver;
  Step 5  repeat for T iterations;
  Step 6  LLM-guided adaptive differentiable projection (hyper-parameter agent LLM_H tunes mu, gamma);
  Step 7  (optional) store the solved instance in D.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..core.agents import AgentError, ChatAgent
from ..llm.base import LLMClient
from .channels import normalize_channels, perturb_csi
from .dataset import MemoryBank
from .metrics import InstanceResult
from .parsing import ALL_KEYS, BF_KEYS, RIS_KEYS, parse_evaluator_feedback, parse_hyperparams, parse_solver_reply
from .physics import STARRISSystem, Solution
from .projection import RoundContext, run_projection
from .prompts import EVALUATOR_NAME, HYPER_NAME, PromptBuilder


def _assemble(blocks: Dict[str, object], M: int, U: int, N: int) -> Solution:
    """Merge the solver blocks into one Solution; missing blocks are zero-filled (only used for display)."""
    wr = blocks.get("wr", np.zeros((M, U)))
    wi = blocks.get("wi", np.zeros((M, U)))
    return Solution.from_arrays(
        wr, wi, blocks.get("alpha", 0.0), blocks.get("a", np.zeros(N)), blocks.get("phi_t", np.zeros(N)), blocks.get("phi_r", np.zeros(N))
    )


class HyperTuner:
    """The hyper-parameter agent LLM_H (eqs. (9)-(11)) wrapped as the projection's ``theta_provider``."""

    def __init__(self, client: LLMClient, prompts: PromptBuilder, cfg: Config, x_block: Optional[str], y_T: Solution):
        self.cfg = cfg
        self.prompts = prompts
        self.x_block = x_block
        self.y_T = y_T
        self.agent = ChatAgent(HYPER_NAME, prompts.hyper_system(), client, parse_retries=cfg.projection.hyper_retries,
                               temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens)
        self.suggestions: List[Dict[str, Any]] = []

    def __call__(self, ctx: RoundContext) -> Tuple[float, float]:
        user = self.prompts.hyper_user(ctx, self.x_block, self.y_T)
        try:
            out, _ = self.agent.ask(user, lambda txt: parse_hyperparams(txt, self.cfg.framework.trace_max_chars), tag=f"l{ctx.round_idx}")
        except AgentError:
            self.suggestions.append(dict(round=ctx.round_idx, failed=True, theta=ctx.theta))
            return ctx.theta  # keep the current hyper-parameters
        self.suggestions.append(dict(round=ctx.round_idx, failed=False, mu=out.mu, gamma=out.gamma))
        return out.mu, out.gamma


class LLM4RRMBeamformer:
    """Multi-agent feedback-driven LLM solver + adaptive projection for one STAR-RIS instance at a time."""

    def __init__(self, cfg: Config, client: LLMClient, memory: Optional[MemoryBank] = None, seed: int = 0):
        self.cfg = cfg
        self.client = client
        self.memory = memory
        self.rng = np.random.default_rng(seed)
        self.prompts = PromptBuilder(cfg.system, cfg.framework, cfg.projection)
        if memory is not None:
            memory.check_compatible(cfg.system)

    # ----------------------------------------------------------------------------------
    def solve(self, G: np.ndarray, H: np.ndarray, index: int = 0, repeat: int = 0, csi_seed: Optional[int] = None) -> InstanceResult:
        cfg, fw, pc, sc = self.cfg, self.cfg.framework, self.cfg.projection, self.cfg.system
        G = np.asarray(G)[None] if np.asarray(G).ndim == 2 else np.asarray(G)
        H = np.asarray(H)[None] if np.asarray(H).ndim == 2 else np.asarray(H)
        sys_true = STARRISSystem(G, H, sc)
        if fw.csi_error_eta > 0.0:
            Ge, He = perturb_csi(G, H, fw.csi_error_eta, sc.noise_power_w, seed=(csi_seed if csi_seed is not None else index))
            sys_est = STARRISSystem(Ge, He, sc)
        else:
            Ge, He, sys_est = G, H, sys_true
        Gn, Hn, _, _ = normalize_channels(Ge, He)
        Gn, Hn = Gn[0], Hn[0]

        usage0 = self.client.usage()
        t_wall0 = time.perf_counter()
        K = fw.eff_num_solvers
        pb = self.prompts
        M, U, N = sc.M, sc.U, sc.N
        x_block = pb.system_parameters(Gn, Hn)

        # ---- Step 1: example selection --------------------------------------------------
        examples = []
        if self.memory is not None and fw.eff_n_fewshot > 0:
            examples = self.memory.retrieve(Gn, Hn, fw.eff_n_fewshot, fw.similarity, self.rng)
        scopes = {k: ("all" if K == 1 else ("bf" if k == 1 else "ris")) for k in range(1, K + 1)}
        solver_keys = {k: (ALL_KEYS if K == 1 else (BF_KEYS if k == 1 else RIS_KEYS)) for k in range(1, K + 1)}
        solvers = {
            k: ChatAgent(pb.solver_name(k), pb.solver_system(k), self.client, parse_retries=fw.parse_retries,
                         temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens)
            for k in range(1, K + 1)
        }
        evaluator = None
        if fw.eff_use_evaluator:
            evaluator = ChatAgent(EVALUATOR_NAME, pb.evaluator_system(), self.client, parse_retries=fw.parse_retries,
                                  temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens)

        # ---- Steps 2-5: multi-agent reasoning loop ---------------------------------------
        blocks_by_agent: Dict[int, Dict[str, object]] = {}
        feedback: List[Optional[str]] = [None] * K
        iterations: List[Dict[str, Any]] = []
        sols: List[Solution] = []
        prev_sr: Optional[float] = None
        stopped_at = fw.T - 1

        for t in range(fw.T):
            traces = [""] * K
            failed = [False] * K
            for k in range(1, K + 1):
                prior = _assemble(blocks_by_agent.get(1, {}), M, U, N) if (K == 2 and k == 2) else None
                if t == 0:
                    fewshot = pb.fewshot_block(examples, scopes[k])
                    user = pb.solver_first_user(k, x_block, fewshot, prior)
                else:
                    user = pb.solver_next_user(k, t, feedback[k - 1], prior)
                try:
                    out, _ = solvers[k].ask(
                        user, lambda txt, keys=solver_keys[k]: parse_solver_reply(txt, keys, M, U, N, fw.trace_max_chars), tag=f"t{t}")
                    blocks_by_agent[k] = out.block
                    traces[k - 1] = out.reasoning
                except AgentError:
                    failed[k - 1] = True  # re-asks exhausted: reuse the previous block (random block at t = 0)
                    if k not in blocks_by_agent:  # nothing to fall back on at t = 0: random block
                        rnd = Solution.random(sc, 1, self.rng).to_dict()
                        blocks_by_agent[k] = _random_block(rnd, solver_keys[k])
            merged: Dict[str, object] = {}
            for k in range(1, K + 1):
                merged.update(blocks_by_agent[k])
            sol_t = _assemble(merged, M, U, N)
            ev_t = sys_est.evaluate(sol_t)
            sols.append(sol_t)
            sr = float(ev_t["sum_rate_mbps"][0])
            iterations.append(dict(
                t=t, sum_rate_mbps=sr, rates_mbps=ev_t["rates_mbps"][0].tolist(), n_violations=int(ev_t["n_violations"][0]),
                feasible=bool(ev_t["feasible"][0]), agent_failed=failed, feedback_received=list(feedback)))

            # stop if constraints satisfied and performance converged, or t = T
            converged = bool(ev_t["feasible"][0]) and prev_sr is not None and abs(sr - prev_sr) <= fw.stop_rel_tol * max(abs(sr), 1e-9)
            prev_sr = sr
            if converged or t == fw.T - 1:
                stopped_at = t
                break

            # ---- Step 4: evaluator feedback ----------------------------------------------
            if evaluator is not None:
                if t == 0:
                    ev_user = pb.evaluator_first_user(x_block, pb.fewshot_block(examples, "all"), sol_t, ev_t, traces)
                else:
                    ev_user = pb.evaluator_next_user(t, sol_t, ev_t, traces)
                fb, _ = evaluator.ask(ev_user, lambda txt: parse_evaluator_feedback(txt, K), tag=f"t{t}")
                feedback = list(fb)
                iterations[-1]["feedback_generated"] = list(fb)

        # y^(T): last iterate (paper) or best iterate so far
        if fw.final_selection == "best":
            def key(i):
                it = iterations[i]
                return (it["n_violations"], -it["sum_rate_mbps"])
            y_T = sols[min(range(len(sols)), key=key)]
        else:
            y_T = sols[-1]
        ev_pre = sys_true.evaluate(y_T)

        # ---- Step 6: LLM-guided adaptive projection ---------------------------------------
        sys_proj = sys_true if fw.projection_uses_true_csi else sys_est
        tuner = None
        if pc.hyper_mode == "llm":
            xb = pb.system_parameters(Gn, Hn) if pc.hyper_include_channels else pb.system_parameters(Gn, Hn, include_channels=False)
            tuner = HyperTuner(self.client, pb, cfg, xb, y_T)
        res = run_projection(sys_proj, y_T, pc, theta_provider=tuner)
        final = res.solution.wrapped()
        ev_post = sys_true.evaluate(final)

        # ---- Step 7: update D -----------------------------------------------------------------
        if fw.update_memory and self.memory is not None and bool(ev_post["feasible"][0]):
            Gt, Ht, _, _ = normalize_channels(G, H)
            self.memory.add(Gt[0], Ht[0], final, float(ev_post["sum_rate_mbps"][0]), ev_post["rates_mbps"][0])

        usage1 = self.client.usage()
        llm_usage = {k: usage1[k] - usage0[k] for k in usage1}
        agents_for_failures = list(solvers.values()) + ([evaluator] if evaluator else []) + ([tuner.agent] if tuner else [])
        transcripts = None
        if cfg.experiment.save_transcripts:
            transcripts = {a.name: a.transcript for a in agents_for_failures}
        return InstanceResult(
            index=index, repeat=repeat, ok=True,
            pre=_summ(ev_pre), post=_summ(ev_post),
            s_zv=res.s_zv, s_conv=res.s_conv, projection_steps=res.steps_taken, n_hyper_calls=res.n_hyper_calls,
            theta_history=[list(th) for th in res.theta_history], projection_trajectory=res.trajectory,
            iterations=iterations, stopped_at_iteration=stopped_at, llm=llm_usage,
            time_llm_s=llm_usage["latency_s"], time_projection_s=res.time_s, time_total_s=time.perf_counter() - t_wall0,
            y_T=y_T.to_dict(), y_final=final.to_dict(),
            n_agent_failures=sum(a.n_failed_parses for a in agents_for_failures),  # unparsable replies (re-asked)
            transcripts=transcripts,
        )


def _summ(ev: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return dict(sum_rate_mbps=float(ev["sum_rate_mbps"][0]), rates_mbps=ev["rates_mbps"][0].tolist(),
                n_violations=int(ev["n_violations"][0]), feasible=bool(ev["feasible"][0]))


def _random_block(rnd: Dict[str, Any], keys: List[str]) -> Dict[str, object]:
    block: Dict[str, object] = {}
    if "wr" in keys:
        block.update(wr=np.array(rnd["Wr"]), wi=np.array(rnd["Wi"]), alpha=float(rnd["alpha"]))
    if "a" in keys:
        block.update(a=np.array(rnd["a"]), phi_t=np.array(rnd["phi_t"]), phi_r=np.array(rnd["phi_r"]))
    return block
