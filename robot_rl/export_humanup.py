"""Export validated Stage I recovery trajectories for HumanUP refinement.

python -m robot_rl.export_humanup --model MODEL.zip --output FILE.npz
Only a pair of successful, physically simulated rollouts is written. No mocap,
nearest-frame bank, or kinematic retarget can silently substitute for discovery.
"""
from pathlib import Path
import argparse
import hashlib

import numpy as np

from robot_rl.humanup import FORMAT, MODEL_CONTRACT
from robot_rl.robot import JOINTS
from robot_rl.recovery import RECOVERY_STEPS


def discover_rollout(model, pose, seed):
    from robot_rl.env import HumanoidBalanceEnv
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        obs, info = env.reset(seed=seed, options={"pose": pose})
        positions, heights = [info["joint_q"].copy()], [info["head_height"]]
        for _ in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            positions.append(info["joint_q"].copy())
            heights.append(info["head_height"])
            if info.get("is_recovered"):
                # Validate five seconds of actual stability, but export only
                # up to its onset plus .2s; do not stretch a long static tail.
                stop = max(2, len(positions) - RECOVERY_STEPS + 10)
                return np.asarray(positions[:stop]), np.asarray(heights[:stop]), 1. / env.sim.control_hz
            if terminated or truncated:
                break
        return None
    finally:
        env.close()


def export(model_path, output, *, slowdown=8., attempts=8):
    from stable_baselines3 import PPO
    if not np.isfinite(slowdown) or slowdown <= 0. or attempts < 1:
        raise ValueError("slowdown and attempts must be positive")
    model_path, output = Path(model_path), Path(output)
    model = PPO.load(str(model_path), device="cpu")
    if (model.observation_space.shape != (384,) or getattr(model, "humanup_stage", None) != "discovery"
            or getattr(model, "humanup_contract", None) != MODEL_CONTRACT):
        raise ValueError("Export requires a HumanUP discovery checkpoint")
    data = {"format": np.array(FORMAT), "joint_names": np.array([j.name for j in JOINTS]),
            "checkpoint_sha256": np.array(hashlib.sha256(model_path.read_bytes()).hexdigest()),
            "slowdown": np.array(float(slowdown))}
    for pose in ("prone", "supine"):
        for seed in range(attempts):
            result = discover_rollout(model, pose, 20_000 + seed)
            if result is not None:
                q, head, dt = result
                data[f"{pose}_q"] = q.astype(np.float32)
                data[f"{pose}_head_height"] = head.astype(np.float32)
                data[f"{pose}_time"] = (np.arange(len(q)) * dt * slowdown).astype(np.float64)
                break
        else:
            raise RuntimeError(f"No validated {pose} recovery in {attempts} trials; no reference was written")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Do not replace an existing reference until both trajectories pass.
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(temporary, **data)
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slowdown", type=float, default=8.)
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()
    path = export(args.model, args.output, slowdown=args.slowdown, attempts=args.attempts)
    print(f"Validated HumanUP reference saved: {path}")


if __name__ == "__main__":
    main()
