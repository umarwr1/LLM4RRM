# LLM4RRM - Use-Case I (STAR-RIS joint beamforming and phase shift)

Reproduction of *LLM4RRM: A Multi-Agent Feedback-Driven LLM Framework for Constrained Resource Allocation in Wireless Networks*
(Algorithm 1) for the beamforming use case.

## What is implemented

| Paper element | Code |
|---|---|
| STAR-RIS MISO system, SINR/rate, transformations `W = W_hat/||W_hat||_F * Sig(alpha) * sqrt(Pmax)`, `beta_t = sqrt(Sig(a))` | `beamforming/physics.py` |
| Rician channels, path loss, CSI error `g_hat = g + Delta` | `beamforming/channels.py` |
| K solver agents (beamforming, STAR-RIS), independent chat histories | `beamforming/framework.py`, `core/agents.py` |
| Evaluator agent with per-agent feedback | `beamforming/framework.py`, `beamforming/prompts.py` |
| Hyper-parameter agent LLM_H tuning `theta = {mu, gamma}` | `beamforming/framework.py` (`HyperTuner`) |
| Surrogate `V` (eq. 24), momentum projection (eq. 25), analytic gradients | `beamforming/physics.py`, `beamforming/projection.py` |
| Few-shot memory bank D, Frobenius-norm retrieval | `beamforming/dataset.py` |
| Prompt levels 1-7 (Table 4), K / evaluator ablation, temperature, backbones, CSI error, theta0 | `config.py`, `configs/experiments/*.yaml` |
| `R_pre/R_post`, `V_pre/V_post`, `S_ZV`, `S_conv`, `T_conv`, `l_avg` | `beamforming/metrics.py` |
| `D_UQ` (eq. 13), `MAD_UQ` (eq. 14) | `core/uq.py`, `scripts/run_uq.py` |

The physics uses NumPy with hand-derived gradients (checked against finite differences in `tests/test_physics.py`); PyTorch is not required.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q                      # 44 tests, no network, no API key
```

## API keys

No key is stored anywhere in this repository. Copy `.env.example` to `.env` (git-ignored) or export the variables:

| provider preset | variable |
|---|---|
| `gpt-5-mini`, `gpt-4.1-mini` | `OPENAI_API_KEY` |
| `grok-4-fast` | `XAI_API_KEY` |
| `gemini-2.5-flash` | `GEMINI_API_KEY` |
| `llama-4-scout` | `LLAMA_API_KEY`, `LLAMA_BASE_URL` |
| `custom` (`--set llm.model=... llm.base_url=...`) | `LLM_API_KEY`, `LLM_BASE_URL` |

`tests/test_config_and_secrets.py` fails if a key-like string is committed. Notebooks are git-ignored on purpose.

## Usage

```bash
# 1. few-shot dataset (no LLM; also built automatically on first run)
python scripts/generate_dataset.py --size 200

# 2. offline pipeline test (mock LLM)
python scripts/run_beamforming.py --provider mock --n-instances 3

# 3. real run (100 channels, paper default; costs API money)
python scripts/run_beamforming.py --provider gpt-5-mini --n-instances 100 \
    --set system.r_min_mbps=5 --set system.p_max_w=1.5

# 4. a table of the paper
python scripts/run_sweep.py configs/experiments/table4_prompt_levels.yaml --provider gpt-5-mini
python scripts/run_sweep.py configs/experiments/table12_ablation.yaml --only "K2$"

# 5. uncertainty quantification (Psi=20, Lambda=10 => 200 multi-agent runs)
python scripts/run_uq.py --provider gpt-5-mini
```

Outputs go to `results/<run>/`: `config.yaml`, `instances.jsonl` (per instance, optionally with transcripts), `summary.json`.
Sweeps: Tables 2a, 2b, 3, 4/5, 6, 7, 8, 9, 12, 13 (`configs/experiments/`).

Cost per instance at defaults: about `K*T = 10` solver calls, up to 4 evaluator calls, up to `L = 5` hyper-parameter calls, i.e. at most about 19 calls.

## Modelling choices

- Rate `= BW * log2(1+SINR)` with BW = 5 MHz (Rmin = 3 Mbps corresponds to 0.6 bps/Hz).
- Geometry: BS to STAR-RIS 200 m, users in a 50 m disc around the STAR-RIS, heights 25/10/1.5 m, half-wavelength arrays, 2 transmission + 2 reflection users, Rician K = 10, `L(d) = 35.6 + 22 log10 d`. Channels are normalised (mean entry magnitude 1) before being shown to the LLM.
- `T = 5` solver iterations; the evaluator runs after every iteration except the last.
- `(Rmin, Pmax)` pairs follow the paper's tables: (3.0, 1.0), (5.0, 1.5), (17.5, 2.0), (20.0, 5.0).
- `S_conv` is the first projection step whose feasible sum-rate reaches 99 % of the final sum-rate.
- CSI-error study: the LLM sees the perturbed CSI; projection and all reported metrics use the true channels.
- Projection: `probe_steps = 10` steps per tuning round before LLM_H is consulted, at most `L = 5` rounds and `I = 200` steps. It returns the best feasible iterate and keeps refining the sum-rate after feasibility until converged.
- Memory bank: random initialisation plus the same projection, seeded differently from the test channels (enforced).
- `mock` provider: deterministic offline backend for testing the pipeline.
- Temperature sweep (Table 13): models that only support the default temperature ignore other values and emit a warning.

## Layout

```
configs/                 default config + one sweep file per paper table
scripts/                 generate_dataset.py, run_beamforming.py, run_sweep.py, run_uq.py
src/llm4rrm/             config, cli, utils, core/, llm/, beamforming/
tests/                   physics, parsing, prompts, dataset, projection, UQ, mock end-to-end, secrets scan
```
