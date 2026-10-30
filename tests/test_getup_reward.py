import numpy as np
import pytest

from robot_rl.env import HumanoidBalanceEnv
from robot_rl.rewards import GetupReward
from robot_rl.humanup import GetupCurriculum


def test_stationary_traps_are_negative_and_standing_rewards_stillness():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        _, stand = env.reset(options={"pose": "stand"})
        floor = np.r_[env.sim.weight / 2, env.sim.weight / 2, np.zeros(13)]
        stand.update(floor_n=floor, foot_contact=(True, True), feet_now=(True, True),
                     upright=1., chest_upright=1., base_lin_vel=np.zeros(3),
                     base_ang_vel=np.zeros(3), foot_slip=0.)
        seated = dict(stand, base_height=.2, floor_n=np.r_[0., .2 * env.sim.weight, .8 * env.sim.weight, np.zeros(12)])
        inverted = dict(stand, upright=-1., chest_upright=-1., head_height=.1)
        _, lying = env.reset(options={"pose": "supine"})
        def stationary(state):
            reward = GetupReward(env.sim.weight)
            reward.prime(state)
            return reward.update(np.zeros(20), state)[0]
        scores = [stationary(s) for s in (stand, seated, inverted, lying)]
        assert scores[0] > 0.
        assert all(value < 0. for value in scores[1:])
        assert stationary(dict(stand, base_lin_vel=np.array([.4, 0., 0.]))) < scores[0]
        print("static stand/seated/inverted/lying:", scores)
    finally:
        env.close()


def test_up_down_cannot_farm_discounted_progress():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        _, lying = env.reset(options={"pose": "supine"})
        raised = dict(lying, base_height=lying["base_height"] + .2, upright=1., chest_upright=1.)
        reward = GetupReward(env.sim.weight)
        reward.prime(lying)
        _, first = reward.update(np.zeros(20), raised)
        _, second = reward.update(np.zeros(20), lying)
        assert first["progress"] > 0.
        assert first["progress"] + .99 * second["progress"] <= 1e-10
    finally:
        env.close()


def test_assistance_is_training_only_and_never_a_policy_input():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        env.reset(seed=1, options={"pose": "supine"})
        before = env._observation().copy()
        env._pose_curriculum.level = 2
        env.assist_fraction = .6
        np.testing.assert_array_equal(before, env._observation())
        env.step(np.zeros(20))
        assert env.assist_fraction == 0.  # explicit test pose
        env._explicit_pose = False
        env._pose_curriculum.level = 0
        env.step(np.zeros(20))
        assert 0. < env.assist_fraction <= .6
        assert env._recovery.stable_steps == 0
        env._steps = env.max_episode_steps - 250
        env.step(np.zeros(20))
        assert env.assist_fraction == 0.
    finally:
        env.close()


def test_assistance_withdrawal_requires_success_and_prone_waits_for_zero_help():
    course = GetupCurriculum()
    for _ in range(100):
        course.update("supine", False)
    assert course.level == 0 and course.assistance == .6
    for _ in range(100):
        course.update("stand", True)
    assert course.level == 0
    for _ in range(40):
        course.update("supine", True)
    assert course.assistance == 0.
    assert course.prone_probability > 0.
