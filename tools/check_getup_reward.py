"""Bounded dynamics audit, not a learning-success test.

Run: python -m tools.check_getup_reward --policy runs/humanoid_ppo_getup.zip
Replays a policy and zero-action baselines without starting training.
"""
import argparse
import json
import numpy as np
from robot_rl.env import HumanoidBalanceEnv
from robot_rl.checkpoint import InferencePolicy


def rollout(pose, policy=None, steps=1000, reward_factory=None):
    env = HumanoidBalanceEnv(curriculum="getup", max_episode_steps=steps)
    obs, info = env.reset(seed=10000, options={"pose": pose})
    audit_reward = reward_factory(env.sim.weight) if reward_factory else None
    if audit_reward:
        audit_reward.prime(info)
    total = 0.
    states = []
    try:
        for _ in range(steps):
            action = np.zeros(20) if policy is None else policy.predict(obs)[0]
            obs, value, terminated, truncated, info = env.step(action)
            if audit_reward:
                value, _ = audit_reward.update(action, info)
            total += value
            states.append([info["base_height"], info["head_height"], info["upright"], info["chest_upright"],
                           float(np.linalg.norm(info["base_lin_vel"])), float(np.linalg.norm(info["base_ang_vel"]))])
            if terminated or truncated:
                break
        return {"return": round(total, 4), "success": bool(info.get("is_success", False)),
                "tail_mean_base_head_up_chest_speed_spin": np.mean(states[-200:], axis=0).round(4).tolist()}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    policy = InferencePolicy(args.policy)
    result = {"zero_action": {}, "policy": {}}
    for pose in ("stand", "supine", "prone"):
        result["zero_action"][pose] = rollout(pose)
        result["policy"][pose] = rollout(pose, policy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
