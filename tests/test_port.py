"""The Python port must reproduce the original MATLAB link (fixtures from make_fixtures.m)."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from radar.config import SounderConfig
from radar.sync import frame_sync, matched_filter
from radar.waveform import lfsr_framesync, preamble_baseband, rrc_pulse

FIXTURE = Path(__file__).parent / "fixtures" / "matlab_reference.mat"


@pytest.fixture(scope="module")
def ref():
    if not FIXTURE.exists():
        pytest.skip("run tests/fixtures/make_fixtures.m in MATLAB first")
    return {k: np.squeeze(v) for k, v in loadmat(FIXTURE).items() if not k.startswith("__")}


def test_pn_sequence_matches(ref):
    np.testing.assert_array_equal(lfsr_framesync(1524), ref["pn"].astype(int))


def test_rrc_pulse_matches(ref):
    np.testing.assert_allclose(rrc_pulse(SounderConfig()), ref["pulse"], atol=1e-12)


def test_preamble_matches(ref):
    np.testing.assert_allclose(preamble_baseband(SounderConfig()), ref["pre"], atol=1e-10)


def test_frame_sync_matches(ref):
    cfg = SounderConfig()
    mf = matched_filter(ref["rx"], rrc_pulse(cfg))
    np.testing.assert_allclose(mf, ref["mf"], atol=1e-9)
    p, dbg = frame_sync(mf, cfg.sc_nsyms, cfg.sc_os)
    assert p == int(ref["bod"]) - 1          # MATLAB is 1-based
    np.testing.assert_allclose(dbg["T_vals"], ref["T_vals"], rtol=1e-6, atol=1e-6)
