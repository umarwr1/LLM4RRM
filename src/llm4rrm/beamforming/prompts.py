"""Prompt construction P_k(.), P_E(.), P_H(.) for the beamforming use case (paper eqs. (3), (7), (10), Appendix).

Agents (names follow the Appendix):
  * solver 1  OPTIBEAM-STAR : active beamforming   y1 = {W_hat, alpha}
  * solver 2  OPTIRIS-STAR  : STAR-RIS             y2 = {a, phi_t, phi_r}
  * K = 1     OPTISOLVE-STAR: all variables
  * evaluator OPTEVAL       : diagnoses the joint solution and writes per-agent feedback
  * hyper     HYPEAGENT     : tunes the projection hyper-parameters theta = {mu, gamma}

Prompt styles (paper Table 4) are cumulative and controlled by ``FrameworkConfig.prompt_level``:
  1 direct instruction + constraints | 2 + few-shot examples | 3 + physics-aware formulation |
  4 + confidence statement | 5 + structured chain-of-thought | 6 + multi-agent decomposition (K = 2) |
  7 + evaluator-guided feedback (full LLM4RRM).

Numbers shown to the LLM use 4 decimals.  Channels are shown normalised by their mean entry magnitude.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import FrameworkConfig, ProjectionConfig, SystemConfig
from .dataset import MemoryEntry
from .physics import Solution

DEC = 4
SOLVER_NAMES = {1: "OPTIBEAM-STAR", 2: "OPTIRIS-STAR"}
SINGLE_SOLVER_NAME = "OPTISOLVE-STAR"
EVALUATOR_NAME = "OPTEVAL"
HYPER_NAME = "HYPEAGENT"


# --------------------------------------------------------------------------------------
# number / matrix formatting
# --------------------------------------------------------------------------------------
def _f(x: float) -> str:
    return f"{float(x):.{DEC}f}"


def _c(z: complex) -> str:
    return f"{z.real:.{DEC}f}{z.imag:+.{DEC}f}j"


def fmt_vec(v: Sequence[float]) -> str:
    return "[" + ", ".join(_f(x) for x in np.asarray(v).reshape(-1)) + "]"


def fmt_rmat(A: np.ndarray) -> str:
    return "[" + ", ".join(fmt_vec(row) for row in np.asarray(A)) + "]"


def fmt_cmat(A: np.ndarray) -> str:
    rows = ["  [" + ", ".join(_c(z) for z in row) + "]" for row in np.asarray(A)]
    return "[\n" + ",\n".join(rows) + "\n]"


def fmt_solution(sol: Solution, scope: str = "all", idx: int = 0) -> str:
    """Solution variables as ``name=value`` lines.  scope: 'bf' (W_hat, alpha) | 'ris' (a, phi) | 'all'."""
    d = sol.wrapped().to_dict(idx, decimals=DEC)
    lines = []
    if scope in ("bf", "all"):
        lines += [f"Wr={fmt_rmat(np.array(d['Wr']))}", f"Wi={fmt_rmat(np.array(d['Wi']))}", f"alpha={_f(d['alpha'])}"]
    if scope in ("ris", "all"):
        lines += [f"a={fmt_vec(d['a'])}", f"phi_t={fmt_vec(d['phi_t'])}", f"phi_r={fmt_vec(d['phi_r'])}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# prompt builder
# --------------------------------------------------------------------------------------
class PromptBuilder:
    def __init__(self, sys_cfg: SystemConfig, fw_cfg: FrameworkConfig, proj_cfg: ProjectionConfig):
        self.sys = sys_cfg
        self.fw = fw_cfg
        self.proj = proj_cfg
        self.K = fw_cfg.eff_num_solvers

    # ------------------------------------------------------------------ shared text --
    def _agent_scope(self, k: int) -> str:
        if self.K == 1:
            return "all"
        return "bf" if k == 1 else "ris"

    def solver_name(self, k: int) -> str:
        return SINGLE_SOLVER_NAME if self.K == 1 else SOLVER_NAMES[k]

    def _problem_text(self) -> str:
        s = self.sys
        lines = [
            "PROBLEM",
            f"A base station (BS) with M antennas serves U single-antenna users through a STAR-RIS with N elements "
            f"(direct BS-user links are blocked). Users 1..Ut (Ut = {s.n_transmit_users}) are in the transmission space of "
            f"the STAR-RIS and users Ut+1..U (Ur = {s.n_reflect_users}) are in its reflection space.",
            "Maximise the network sum-rate sum_u R_u subject to:",
            f"  (a) QoS: R_u >= Rmin = {_f(s.r_min_mbps)} Mbps for every user u,",
            f"  (b) power: sum_u ||w_u||^2 <= Pmax = {_f(s.p_max_w)} W  (and energy splitting beta_t^2 + beta_r^2 = 1 per element).",
        ]
        if self.fw.style_physics:
            lines += [
                "",
                "PHYSICS",
                "Received signal of user u: y_u = h_u^H Theta_l G sum_i w_i s_i + z_u, z_u ~ CN(0, sigma^2), where l = t for "
                "transmission users and l = r for reflection users, and Theta_l = diag(beta_{1,l} e^{j phi_{1,l}}, ..., beta_{N,l} e^{j phi_{N,l}}).",
                "SINR_u = |h_u^H Theta_l G w_u|^2 / ( sum_{i != u} |h_u^H Theta_l G w_i|^2 + sigma^2 ),   R_u = BW * log2(1 + SINR_u).",
                "G in C^{N x M} is the BS->STAR-RIS channel, h_u in C^{N x 1} (column u of H in C^{N x U}) is the STAR-RIS->user-u channel.",
                "The noise power is tiny, so the system is interference-limited: prefer (nearly) orthogonal effective channels "
                "c_u = h_u^H Theta_l G.",
            ]
        return "\n".join(lines)

    def _transform_text(self, scope: str) -> str:
        s = self.sys
        lines = ["TRANSFORMATIONS (constraints are enforced automatically from your unconstrained variables)"]
        if scope in ("bf", "all"):
            lines.append(
                f"  W = W_hat / ||W_hat||_F * Sig(alpha) * sqrt(Pmax),  W_hat = Wr + j*Wi in C^(M x U), Sig(x) = 1/(1+exp(-x));  "
                f"this guarantees sum_u ||w_u||^2 <= Pmax = {_f(s.p_max_w)}."
            )
        if scope in ("ris", "all"):
            lines.append(
                "  beta_t,n = sqrt(Sig(a_n)),  beta_r,n = sqrt(1 - Sig(a_n)),  a_n in R unconstrained;  "
                "this guarantees beta_t,n^2 + beta_r,n^2 = 1. phi_t, phi_r are phases in [0, 2*pi)."
            )
        if scope == "bf" and self.K == 2:
            lines.append("  The STAR-RIS variables (a, phi_t, phi_r) are optimised by the other solver agent.")
        if scope == "ris" and self.K == 2:
            lines.append("  The beamformer (W_hat, alpha) is optimised by the other solver agent and is given to you as input.")
        return "\n".join(lines)

    def _strategy_text(self, scope: str) -> str:
        if not self.fw.style_physics:
            return ""
        lines = ["STRATEGY HINTS"]
        if scope in ("bf", "all"):
            lines += [
                "  - Beamformer: align w_u with the effective channel c_u^H (maximum-ratio direction) while keeping the leakage "
                "|c_u w_i|^2, i != u, small (zero-forcing-like); distribute power so that weak users still reach Rmin.",
                "  - alpha sets the used fraction of Pmax: Sig(alpha) close to 1 means (almost) full power.",
            ]
        if scope in ("ris", "all"):
            lines += [
                "  - Phases: for user u on side l, the cascaded paths add coherently when phi_{n,l} = -arg( conj(h_{n,u}) * [G w_u]_n ) "
                "(use the dominant users of each side; phases are periodic with 2*pi).",
                "  - Amplitudes: Sig(a_n) is the fraction of the incident energy transmitted (the rest is reflected); give more "
                "to the side that hosts the weaker users.",
            ]
        return "\n".join(lines)

    def _style_suffix(self) -> str:
        parts = []
        if self.fw.style_cot:
            parts.append(
                "Reason step by step: (1) analyse the channel matrices and identify dominant paths, (2) choose the variables "
                "of your block, (3) check the constraints and the expected interference, (4) state the final values."
            )
        else:
            parts.append("Keep the reasoning short.")
        if self.fw.style_confidence:
            parts.append("End your reasoning with one sentence stating your confidence in the solution's quality.")
        return " ".join(parts)

    def _solver_output_format(self, scope: str) -> str:
        s = self.sys
        fields = []
        if scope in ("bf", "all"):
            fields += [
                f'  "Wr": [[...], ...]   (M x U = {s.M} x {s.U}: {s.M} rows with {s.U} real numbers; real part of W_hat)',
                f'  "Wi": [[...], ...]   ({s.M} x {s.U}; imaginary part of W_hat)',
                '  "alpha": number       (real scalar)',
            ]
        if scope in ("ris", "all"):
            fields += [
                f'  "a": [...]            ({s.N} real numbers, unconstrained)',
                f'  "phi_t": [...]        ({s.N} real numbers in [0, 2*pi])',
                f'  "phi_r": [...]        ({s.N} real numbers in [0, 2*pi])',
            ]
        return (
            "OUTPUT FORMAT\nFirst give your reasoning, then end your reply with exactly one JSON object inside a ```json code block "
            "with these fields:\n{\n" + ",\n".join(fields) + "\n}"
        )

    # ------------------------------------------------------------------ system params --
    def system_parameters(self, G_n: np.ndarray, H_n: np.ndarray, include_channels: bool = True) -> str:
        s = self.sys
        lines = [
            "SYSTEM PARAMETERS x",
            f"- Dimensions: M = {s.M} BS antennas, N = {s.N} STAR-RIS elements, U = {s.U} users "
            f"(Ut = {s.n_transmit_users} transmission-space users: indices 1..{s.n_transmit_users}; "
            f"Ur = {s.n_reflect_users} reflection-space users).",
            f"- Pmax = {_f(s.p_max_w)} W; QoS Rmin = {_f(s.r_min_mbps)} Mbps per user "
            f"(= {_f(s.r_min_bps_hz)} bps/Hz at bandwidth BW = {s.bandwidth_hz / 1e6:g} MHz); "
            f"noise power sigma^2 = {s.noise_power_w:.3e} W ({s.noise_power_dbm:g} dBm).",
        ]
        if include_channels:
            lines += [
                f"- G (BS->STAR-RIS), shape N x M = {s.N} x {s.M} (rows: STAR-RIS elements, columns: BS antennas):",
                fmt_cmat(G_n),
                f"- H = [h_1, ..., h_U] (STAR-RIS->users), shape N x U = {s.N} x {s.U} (column u is h_u):",
                fmt_cmat(H_n),
                "(Channels are shown normalised by their mean entry magnitude so that the numbers are O(1); the rates reported "
                "to you are computed with the true channels, and the decision variables are invariant to this scaling.)",
            ]
        return "\n".join(lines)

    def fewshot_block(self, entries: Sequence[MemoryEntry], scope: str) -> str:
        if not entries:
            return ""
        lines = ["REFERENCE EXAMPLES (past instances solved with ALL QoS constraints satisfied; use them to infer how channels map to good solutions)"]
        for i, e in enumerate(entries, 1):
            lines += [
                f"Example {i}: G =", fmt_cmat(e.G), "H =", fmt_cmat(e.H),
                f"best solution found (sum-rate = {_f(e.sum_rate_mbps)} Mbps, all users >= Rmin):",
                fmt_solution(e.solution, scope),
                "",
            ]
        return "\n".join(lines).rstrip()

    def performance_block(self, ev: Dict[str, np.ndarray], title: str = "PERFORMANCE of the joint solution (computed with the system model)") -> str:
        s = self.sys
        rates = ev["rates_mbps"][0]
        viol = ev["violated"][0]
        parts = []
        for u, (r, v) in enumerate(zip(rates, viol), 1):
            parts.append(f"R_{u} = {_f(r)} Mbps [{'VIOLATED' if v else 'OK'}]")
        flags = [0 if v else 1 for v in viol]
        return "\n".join([
            title + ":",
            f"- sum-rate F = {_f(ev['sum_rate_mbps'][0])} Mbps",
            "- " + "; ".join(parts) + f"   (Rmin = {_f(s.r_min_mbps)} Mbps)",
            f"- constraint flags f = {flags} (1 = satisfied, 0 = violated); violated users: {int(viol.sum())}/{s.U}",
        ])

    # ------------------------------------------------------------------ solvers --------
    def solver_system(self, k: int) -> str:
        scope = self._agent_scope(k)
        name = self.solver_name(k)
        if self.K == 1:
            role = (f"You are {name}, an expert in wireless communications, beamforming and STAR-RIS design for MISO downlink systems. "
                    "You design the complete network configuration (active beamforming and STAR-RIS).")
        elif k == 1:
            role = (f"You are {name}, an expert in wireless communications and beamforming for STAR-RIS-assisted MISO downlink systems. "
                    "You are solver agent 1 of 2: you design the ACTIVE BEAMFORMING (W_hat, alpha). Solver agent 2 (OPTIRIS-STAR) designs the "
                    "STAR-RIS. " + ("An evaluator agent critiques the joint solution and sends you feedback." if self.fw.eff_use_evaluator else ""))
        else:
            role = (f"You are {name}, an expert in STAR-RIS phase-shift and amplitude optimisation for MISO downlink systems. "
                    "You are solver agent 2 of 2: you design the STAR-RIS (a, phi_t, phi_r) given the beamformer produced by solver agent 1 "
                    "(OPTIBEAM-STAR). " + ("An evaluator agent critiques the joint solution and sends you feedback." if self.fw.eff_use_evaluator else ""))
        parts = [role.strip(), self._problem_text(), self._transform_text(scope)]
        strat = self._strategy_text(scope)
        if strat:
            parts.append(strat)
        parts += [self._solver_output_format(scope), self._style_suffix()]
        return "\n\n".join(p for p in parts if p)

    def solver_first_user(self, k: int, x_block: str, fewshot: str, prior: Optional[Solution]) -> str:
        """Prompt Pi_k^(0): x, S_M, (outputs of the preceding solvers)."""
        parts = [x_block]
        if fewshot:
            parts.append(fewshot)
        if prior is not None and self.K == 2 and k == 2:
            parts.append("OUTPUT OF SOLVER AGENT 1 (fixed input for you, iteration 1):\n" + fmt_solution(prior, "bf"))
        parts.append(f"ITERATION 1 of {self.fw.T}. Produce your first solution now.")
        return "\n\n".join(parts)

    def solver_next_user(self, k: int, t: int, feedback: Optional[str], prior: Optional[Solution]) -> str:
        """Prompt Pi_k^(t), t >= 1: evaluator feedback Delta_k^(t) + preceding solvers' outputs of this iteration."""
        parts = [f"ITERATION {t + 1} of {self.fw.T}."]
        if feedback:
            parts.append(f"FEEDBACK from the evaluator for your block:\n{feedback}")
        else:
            parts.append("No feedback is available. Reconsider your previous solution and improve it.")
        if prior is not None and self.K == 2 and k == 2:
            parts.append("UPDATED OUTPUT OF SOLVER AGENT 1 for this iteration (fixed input for you):\n" + fmt_solution(prior, "bf"))
        parts.append("Provide an improved solution that maximises the sum-rate while satisfying every QoS constraint. "
                     "Reply in the same format (reasoning, then one ```json block).")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ evaluator ------
    def evaluator_system(self) -> str:
        s = self.sys
        K = self.K
        agents = ("one solver agent (OPTISOLVE-STAR) that designs all variables" if K == 1
                  else "two solver agents: OPTIBEAM-STAR (active beamforming: W_hat, alpha) and OPTIRIS-STAR (STAR-RIS: a, phi_t, phi_r)")
        fmt_tags = "\n".join(f"<FEEDBACK_AGENT_{k}>\n...\n</FEEDBACK_AGENT_{k}>" for k in range(1, K + 1))
        return "\n\n".join([
            f"You are {EVALUATOR_NAME}, the expert assistant for STAR-RIS beamforming optimisation. You analyse STAR-RIS-assisted MISO "
            f"systems to identify constraint violations and generate corrective feedback. You supervise {K} solver agent(s): {agents}.",
            "MATHEMATICAL FRAMEWORK\n"
            "R = sum_u R_u,  R_u = BW*log2(1 + gamma_u),  gamma_u = |h_u^H Theta_l G w_u|^2 / ( sum_{i != u} |h_u^H Theta_l G w_i|^2 + sigma^2 ),\n"
            f"QoS: R_u >= Rmin = {_f(s.r_min_mbps)} Mbps for all u.  W = W_hat/||W_hat||_F * Sig(alpha) * sqrt(Pmax),  "
            "beta_t = sqrt(Sig(a)), beta_r = sqrt(1 - Sig(a)),  Theta_l = diag(beta_l e^{j phi_l}).",
            "RESPONSE FORMAT\n"
            "1) <THOUGHT PROCESS>: analyse the variable configurations, user rates and violations; identify what is effective or "
            "problematic (e.g. power imbalance, phase misalignment, inter-user interference). Check the reasoning trace of each solver agent.\n"
            f"2) FEEDBACK: give a short, direct, actionable instruction (at most 5 lines) for each solver agent, saying which variables to "
            f"change and in which direction. Do NOT restate variable values, rates or channel data. Use exactly this tagged format:\n{fmt_tags}\n"
            f'If a block needs no change write "{_NO_CHANGE_TEXT}" inside its tags.',
        ])

    def evaluator_first_user(self, x_block: str, fewshot: str, sol: Solution, ev: Dict[str, np.ndarray],
                             traces: List[str]) -> str:
        parts = [x_block]
        if fewshot:
            parts.append(fewshot)
        parts.append(self._eval_state(0, sol, ev, traces))
        return "\n\n".join(parts)

    def evaluator_next_user(self, t: int, sol: Solution, ev: Dict[str, np.ndarray], traces: List[str]) -> str:
        return ("The solver agents applied your previous feedback. Here is their updated joint solution.\n\n"
                + self._eval_state(t, sol, ev, traces))

    def _eval_state(self, t: int, sol: Solution, ev: Dict[str, np.ndarray], traces: List[str]) -> str:
        parts = [f"ITERATION {t + 1}: JOINT SOLUTION y_hat^({t})", fmt_solution(sol, "all"), self.performance_block(ev)]
        for k, tr in enumerate(traces, 1):
            parts.append(f"REASONING TRACE of solver agent {k} ({self.solver_name(k)}):\n{tr if tr else '(none provided)'}")
        parts.append("Now write your <THOUGHT PROCESS> and the tagged FEEDBACK for each solver agent.")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ hyper agent -----
    def hyper_system(self) -> str:
        p, s = self.proj, self.sys
        return "\n\n".join([
            f"You are {HYPER_NAME}, the adaptive controller responsible for dynamically selecting the projection hyper-parameters "
            "theta = {mu, gamma} that enforce feasibility (all QoS constraints R_u >= Rmin) of a STAR-RIS beamforming solution via "
            "momentum gradient descent on the surrogate",
            "V(x, y) = (1 - s) * sum_u ReLU( exp(Rmin - R_u) - 1 ) + s * sum_u ( exp(Rmin - R_u) - 1 ),\n"
            "with s = 1 if every user satisfies R_u >= Rmin and s = 0 otherwise. Rates R_u and Rmin are in bps/Hz in this formula.\n"
            "Update rule: y[i+1] = y[i] - mu * v[i+1],  v[i+1] = gamma * v[i] + (1 - gamma) * grad V(y[i]),  v[0] = 0. "
            "y collects the unconstrained variables (W_hat, alpha, a, phi_t, phi_r). Once all users satisfy Rmin the second term keeps "
            "increasing the sum-rate.",
            "Your goal is to reach the feasible region quickly (few projection steps) while avoiding oscillations, and to keep the "
            f"sum-rate high. Allowed ranges: mu in [{p.mu_bounds[0]:g}, {p.mu_bounds[1]:g}], gamma in [{p.gamma_bounds[0]:g}, {p.gamma_bounds[1]:g}].",
            "OBJECTIVES\n"
            "- Analyse the projection behaviour from the step records sp^(l): is the convergence stable, oscillatory or slow? "
            "(a monotonically decreasing V with violations falling is stable; flat V means mu is too small; growing/oscillating V means mu "
            "is too large or gamma too high.)\n"
            "- Suggest updated {mu, gamma} that remove oscillations / accelerate convergence while preserving R_u >= Rmin.",
            "RESPONSE FORMAT\n1) <THOUGHT PROCESS>: describe the observed convergence behaviour and justify the change.\n"
            '2) <UPDATED HYPERPARAMETERS>: end with exactly one JSON object in a ```json block: {"mu": value, "gamma": value}',
        ])

    def hyper_user(self, ctx, x_block: Optional[str], y_T: Optional[Solution]) -> str:
        """Prompt Pi_H^(l)."""
        s = self.sys
        parts: List[str] = []
        if ctx.round_idx == 0:
            if x_block:
                parts.append(x_block)
            if y_T is not None:
                parts.append("CURRENT SOLUTION y_hat^(T) (initial point y[0] of the projection):\n" + fmt_solution(y_T, "all"))
        mu, gamma = ctx.theta
        parts.append(f"ROUND l = {ctx.round_idx}: current hyper-parameters theta^({ctx.round_idx}): mu = {mu:.6g}, gamma = {gamma:.6g}.")
        parts.append(self._records_text(ctx))
        parts.append(f"Remaining projection budget: {ctx.steps_left} steps; QoS is still violated. "
                     f"Provide theta^({ctx.round_idx + 1}).")
        return "\n\n".join(parts)

    def _records_text(self, ctx) -> str:
        s = self.sys
        recs = ctx.records
        n = self.proj.max_records_in_prompt
        if len(recs) > n:
            idx = sorted(set(np.linspace(0, len(recs) - 1, n).round().astype(int).tolist()))
            recs = [recs[i] for i in idx]
        lines = [f"PROJECTION STEP RECORDS sp^({ctx.round_idx}) ({len(ctx.records)} records; rates in Mbps, Rmin = {_f(s.r_min_mbps)} Mbps):"]
        for r in recs:
            lines.append(f"  step {r['step']}: V = {r['V']:.5g}, sum-rate = {r['sum_rate_mbps']:.3f}, min-rate = {r['min_rate_mbps']:.3f}, "
                         f"violations = {r['n_viol']}/{s.U}")
        first, last = ctx.records[0], ctx.records[-1]
        lines.append(f"  summary: V {first['V']:.5g} -> {last['V']:.5g}, violations {first['n_viol']} -> {last['n_viol']}, "
                     f"min-rate {first['min_rate_mbps']:.3f} -> {last['min_rate_mbps']:.3f} Mbps")
        lines.append("Intermediate projected values y[i] at the end of this round:\n" + fmt_solution(ctx.solution, "all"))
        return "\n".join(lines)


_NO_CHANGE_TEXT = "No major changes needed."
