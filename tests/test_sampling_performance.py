import numpy as np

from robot_rl.env import HumanoidBalanceEnv
from robot_rl.sim import HumanoidSim


def test_batched_parking_tracks_active_balls_and_reset():
    sim = HumanoidSim()
    for ball in sim._balls:
        sim.data.qpos[ball.qpos_adr:ball.qpos_adr + 3] += 10.
    sim._pin_inactive()
    for ball in sim._balls:
        np.testing.assert_array_equal(sim.data.qpos[ball.qpos_adr:ball.qpos_adr + 3], ball.park)
    for ball in sim._balls:
        sim._activate(ball, np.array([1., 2., 3.]), np.array([4., 5., 6.]))
    qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
    sim._pin_inactive()
    np.testing.assert_array_equal(sim.data.qpos, qpos)
    np.testing.assert_array_equal(sim.data.qvel, qvel)
    sim._deactivate(sim._balls[0])
    sim._pin_inactive()
    for ball in sim._balls[1:]:
        np.testing.assert_array_equal(sim.data.qpos[ball.qpos_adr:ball.qpos_adr + 3], [1., 2., 3.])
    sim.reset()
    sim._pin_inactive()
    for ball in sim._balls:
        assert not ball.active
        np.testing.assert_array_equal(sim.data.qpos[ball.qpos_adr:ball.qpos_adr + 3], ball.park)


def test_compact_training_infos_keep_rewards_episode_and_terminal_state():
    from stable_baselines3.common.monitor import Monitor
    from robot_rl.vec import _step_all, TRAIN_INFO_KEYS
    env = Monitor(HumanoidBalanceEnv(curriculum="getup", max_episode_steps=2), info_keywords=("is_success",))
    try:
        env.reset(seed=0)
        _, _, done, infos = _step_all([env], np.zeros((1, 20)), compact_infos=True)
        assert not done[0]
        assert "reward_terms" in infos[0] and "regularization_strength" in infos[0]
        assert set(infos[0]) <= TRAIN_INFO_KEYS
        assert "joint_velocity" not in infos[0]
        obs, _, done, infos = _step_all([env], np.zeros((1, 20)), compact_infos=True)
        assert done[0]
        assert "episode" in infos[0] and "terminal_observation" in infos[0]
        assert "is_success" in infos[0] and obs.shape == (1, 384)
        assert infos[0]["TimeLimit.truncated"] is True
    finally:
        env.close()


def test_compact_infos_cross_worker_pipe_with_episode_and_terminal_observation():
    from robot_rl.vec import ShardedVecEnv
    vec = ShardedVecEnv(2, "getup", n_workers=1, compact_infos=True,
                        env_kwargs={"max_episode_steps": 2})
    try:
        assert vec.reset().shape == (2, 384)
        _, _, done, infos = vec.step(np.zeros((2, 20)))
        assert not done.any()
        assert all("joint_velocity" not in info and "reward_terms" in info for info in infos)
        _, _, done, infos = vec.step(np.zeros((2, 20)))
        assert done.all()
        assert all("episode" in info and info["terminal_observation"].shape == (384,) for info in infos)
    finally:
        vec.close()


def test_humanup_regularization_uses_global_terminal_height_across_workers(tmp_path):
    from robot_rl.robot import JOINTS
    from robot_rl.humanup import FORMAT
    reference = tmp_path / "reference.npz"
    q = np.array([j.default for j in JOINTS])
    data = {"format": np.array(FORMAT), "joint_names": np.array([j.name for j in JOINTS])}
    for pose in ("supine", "prone"):
        data.update({f"{pose}_time": np.array([0., 1.]), f"{pose}_q": np.stack([q, q]), f"{pose}_head_height": np.array([1., 1.])})
    np.savez(reference, **data)
    from robot_rl.vec import ShardedVecEnv
    vec = ShardedVecEnv(2, "getup", n_workers=2, compact_infos=True,
                        env_kwargs={"max_episode_steps": 1, "humanup_stage": "refinement", "reference_path": str(reference)})
    try:
        vec.reset()
        # Every worker must receive the scale derived from the global mean.
        _, _, done, infos = vec.step(np.zeros((2, 20)))
        assert done.all()
        heights = [info["termination_height"] for info in infos]
        from robot_rl.rewards import regularization_scale
        expected = regularization_scale(1., np.mean(heights))
        assert vec._regularization_strength == expected
        _, _, _, infos = vec.step(np.zeros((2, 20)))
        assert [info["regularization_strength"] for info in infos] == [expected, expected]
    finally:
        vec.close()
