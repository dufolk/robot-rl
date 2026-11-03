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


def test_assistance_withdrawal_uses_reached_upright_and_mixes_prone_from_start():
    course = GetupCurriculum()
    poses = {course.sample(np.random.default_rng(i)) for i in range(200)}
    assert {"supine", "prone", "random"} <= poses
    for _ in range(100):
        course.update("supine", False)
    assert course.level == 0 and course.assistance == .6
    for _ in range(100):
        course.update("stand", True)
    assert course.level == 0
    for _ in range(200):
        course.update("prone", True)
    assert course.assistance == 0.


def test_assistance_stops_after_assist_window():
    from robot_rl.env import ASSIST_STEPS
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        env.reset(seed=1, options={"pose": "supine"})
        env._explicit_pose = False
        env.step(np.zeros(20))
        assert env.assist_fraction > 0.
        env._steps = ASSIST_STEPS
        env.step(np.zeros(20))
        assert env.assist_fraction == 0.
    finally:
        env.close()


def test_standing_posture_rewards_default_stance_over_raised_arms():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        _, info = env.reset(options={"pose": "stand"})
        info.update(floor_n=np.r_[env.sim.weight/2, env.sim.weight/2, np.zeros(13)],
                    foot_contact=(True, True), feet_now=(True, True), foot_slip=0.,
                    base_lin_vel=np.zeros(3), base_ang_vel=np.zeros(3), upright=1., chest_upright=1.)
        arms = np.array(info["joint_pos"], copy=True)
        arms[14:] = 1.4
        def total(state):
            reward = GetupReward(env.sim.weight)
            reward.prime(state)
            return reward.update(np.zeros(20), state)[1]
        neat, messy = total(info), total(dict(info, joint_pos=arms))
        wide = total(dict(info, feet_distance=.9))
        assert neat["posture"] > 1.5 * .02 and messy["posture"] < .1 * neat["posture"]
        assert wide["feet_distance"] < neat["feet_distance"]
        assert neat["total"] > messy["total"] and neat["total"] > wide["total"]
    finally:
        env.close()


def test_raising_inverted_pelvis_never_reduces_height_deficit():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        _, info = env.reset(options={"pose": "supine"})
        reward = GetupReward(env.sim.weight)
        def terms(**values):
            state = dict(info, **values)
            reward.reset()
            reward.prime(state)
            return reward.update(np.zeros(20), state)[1]
        low = terms(base_height=.2, head_height=.1, upright=-.5, chest_upright=-.75)
        high = terms(base_height=1.1, head_height=.1, upright=-.5, chest_upright=-.75)
        assert high["height_deficit"] == low["height_deficit"]
        assert high["total"] == pytest.approx(low["total"])
        for up in (-1., -.5, 0., .5, 1.):
            quiet = terms(upright=up, chest_upright=up, base_lin_vel=np.zeros(3), base_ang_vel=np.zeros(3))
            moving = terms(upright=up, chest_upright=up, base_lin_vel=np.array([2.,0.,0.]), base_ang_vel=np.array([6.,0.,0.]))
            assert moving["lin_vel"] < quiet["lin_vel"]
            assert moving["ang_vel"] < quiet["ang_vel"]
            assert moving["total"] < quiet["total"]
    finally:
        env.close()


def test_continuous_recovery_order_and_dense_standing_signal_before_stillness():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        _, info = env.reset(options={"pose": "stand"})
        info.update(floor_n=np.r_[env.sim.weight/2, env.sim.weight/2, np.zeros(13)],
                    foot_contact=(True,True), feet_now=(True,True), foot_slip=0.,
                    base_lin_vel=np.zeros(3), base_ang_vel=np.zeros(3))
        scores=[]
        for fraction in (.1, .3, .5, .7, .9, 1.):
            state=dict(info, base_height=.15+fraction*(info["stand_base_height"]-.15),
                       head_height=.1+fraction*(info["stand_head_height"]-.1), upright=1., chest_upright=1.)
            reward=GetupReward(env.sim.weight)
            reward.prime(state)
            scores.append(reward.update(np.zeros(20), state)[0])
        assert np.all(np.diff(scores)>0.)
        state=dict(info, upright=1., chest_upright=1., base_lin_vel=np.array([.4,0.,0.]))
        reward=GetupReward(env.sim.weight)
        reward.prime(state)
        _, terms=reward.update(np.zeros(20),state)
        assert terms["standing"] > 0.  # no all-or-nothing zero-speed gate
        assert terms["stable"] == 0.
    finally:
        env.close()
