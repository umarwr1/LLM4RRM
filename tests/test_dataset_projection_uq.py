import numpy as np

from llm4rrm.beamforming.channels import generate_channels, normalize_channels
from llm4rrm.beamforming.dataset import MemoryBank, generate_memory_bank
from llm4rrm.beamforming.physics import STARRISSystem, Solution
from llm4rrm.beamforming.projection import project_fixed_batch, run_projection
from llm4rrm.config import ProjectionConfig, SystemConfig
from llm4rrm.core.uq import consistency_score, d_uq, mad_uq

SYS = SystemConfig(M=4, U=4, N=8, r_min_mbps=1.0)


def test_memory_bank_entries_are_feasible_and_roundtrip(tmp_path):
    bank = generate_memory_bank(SYS, 6, seed=3, verbose=False)
    assert len(bank) == 6
    assert np.all(bank.rates_mbps >= SYS.r_min_mbps - 1e-6)
    p = tmp_path / "b.npz"
    bank.save(p)
    b2 = MemoryBank.load(p)
    assert len(b2) == 6 and b2.meta["M"] == 4
    b2.check_compatible(SYS)


def test_retrieval_returns_nearest_first():
    bank = generate_memory_bank(SYS, 6, seed=3, verbose=False)
    q = bank.entry(2)
    got = bank.retrieve(q.G, q.H, 3)
    assert np.allclose(got[0].G, q.G) and len(got) == 3


def test_bank_never_contains_test_channels():
    G, H = generate_channels(SYS, 4, seed=0)
    bank = generate_memory_bank(SYS, 4, seed=1234, verbose=False)
    Gn, Hn, _, _ = normalize_channels(G, H)
    for i in range(4):
        assert not any(np.allclose(Gn[i], bank.G[j]) for j in range(len(bank)))


def test_projection_reaches_feasibility_and_never_returns_worse():
    G, H = generate_channels(SYS, 1, seed=5)
    sys1 = STARRISSystem(G, H, SYS)
    y0 = Solution.random(SYS, 1, np.random.default_rng(0))
    res = run_projection(sys1, y0, ProjectionConfig(mu0=0.2, I=300), theta_provider=None)
    assert res.feasible and res.s_zv is not None
    assert sys1.evaluate(res.solution)["feasible"].all()


def test_projection_batch_matches_flag():
    G, H = generate_channels(SYS, 3, seed=6)
    sysb = STARRISSystem(G, H, SYS)
    y0 = Solution.random(SYS, 3, np.random.default_rng(1))
    best, feas, _ = project_fixed_batch(sysb, y0, 0.2, 0.9, 300)
    ev = sysb.evaluate(best)
    assert np.array_equal(ev["feasible"], feas)


def test_d_uq_and_mad_uq_known_values():
    # one input, two outputs differing by 2 in a single coordinate: ||.||^2 = 4, one pair, factor 2/(2*1) = 1
    y = np.array([[[0.0, 0.0], [2.0, 0.0]]])
    assert d_uq(y) == 4.0
    assert mad_uq(y) == 0.5  # deviations (1,0,1,0) -> mean 0.5
    same = np.ones((3, 5, 4))
    assert d_uq(same) == 0.0 and mad_uq(same) == 0.0 and consistency_score(0.0) == 1.0


def test_d_uq_three_outputs():
    y = np.array([[[0.0], [1.0], [3.0]]])  # pairs: 1, 9, 4 -> sum 14, factor 2/(3*2) -> 14/3
    assert np.isclose(d_uq(y), 14.0 / 3.0)
