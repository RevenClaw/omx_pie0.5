import numpy as np

from conftest import load_script
from omx_pi05 import constants as C


def test_chunks_are_safe_and_smooth():
    server = load_script("fake_policy_server")
    lo, hi = C.joint_limits_array(C.JOINT_LIMIT_MARGIN)
    max_vel, fps = 0.2, C.FPS
    # Start far from HOME (the torque-off exit pose, j2 outside the limit) and keep feeding the last action back.
    state = np.array([*C.MIN_STRESS_EXIT, 0.0])
    state[1] = -1.839
    prev = np.clip(state[:4], lo, hi)
    for i in range(200):
        chunk = server.make_chunk(state, i * C.EXECUTE_STEPS / fps, C.ACTION_HORIZON, fps, max_vel)
        assert chunk.shape == (C.ACTION_HORIZON, C.ACTION_DIM)
        assert np.all(chunk[:, :4] >= lo - 1e-9) and np.all(chunk[:, :4] <= hi + 1e-9)
        steps = np.abs(np.diff(np.vstack([prev, chunk[:, :4]]), axis=0))
        assert np.all(steps <= max_vel / fps + 1e-9)
        assert set(np.unique(chunk[:, 4])) <= {0.0, 1.0}
        state = np.append(chunk[C.EXECUTE_STEPS - 1, :4], chunk[C.EXECUTE_STEPS - 1, 4])
        prev = state[:4]
