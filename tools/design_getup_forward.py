"""Forward-only get-up design experiments. No autograd or pretrained policy."""
from pathlib import Path
import argparse
import json
import time

import mujoco
import numpy as np

from robot_rl.sim import HumanoidSim


def joint_pose(values, sim):
    """Symmetric waist, hip, knee, ankle, shoulder, elbow, arm spread."""
    waist, hip, knee, ankle, shoulder, elbow, spread = values
    q = sim.default_q.copy()
    q[1] = waist
    q[[4, 10]] = hip
    q[[5, 11]] = knee
    q[[6, 12]] = ankle
    q[[14, 17]] = shoulder
    q[[16, 19]] = elbow
    q[[15, 18]] = [-spread, spread]
    return np.clip(q, sim.q_low, sim.q_high)


def fast_step(sim, target):
    sim.q_target[:] = target
    for _ in range(sim.decimation):
        sim._apply_pd()
        mujoco.mj_step(sim.model, sim.data)


def metrics(sim):
    r = sim.data.xmat[sim.pelvis_id].reshape(3, 3)
    chest = sim.data.geom_xmat[sim.chest_geom_id].reshape(3, 3)
    head = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, 'head')
    return np.array([sim.data.xpos[sim.pelvis_id, 2], sim.data.geom_xpos[head, 2],
                     r[2, 2], chest[2, 2], np.linalg.norm(sim.data.qvel[:3]),
                     np.linalg.norm(sim.data.qvel[3:6])])


DEFAULT = np.array([
    [.65, -1.5, 2.15, -.65, 1.0, .15, .2],
    [.4, -1.4, 2.1, -.65, .6, .1, .2],
    [.35, -1.1, 1.8, -.7, -.6, .4, .2],
    [.15, -.65, 1.2, -.5, -.35, .55, .25],
])


def trajectory(sim, poses, durations, start='supine', keep=False):
    sim.reset(start)
    initial = sim.default_q.copy()
    waypoints = [joint_pose(v, sim) for v in poses] + [sim.default_q.copy()]
    rows, qs, states = [], [], []
    for target, duration in zip(waypoints, [*durations, 2.]):
        for k in range(max(1, round(duration * 50))):
            a = min(1., (k + 1) / max(1., duration * 50))
            a = a * a * (3 - 2 * a)
            q = initial * (1 - a) + target * a
            fast_step(sim, q)
            rows.append(metrics(sim))
            if keep:
                qs.append(q.copy())
                states.append(sim.data.qpos[:27].copy())
        initial = target.copy()
    for _ in range(250):
        fast_step(sim, initial)
        rows.append(metrics(sim))
        if keep:
            qs.append(initial.copy())
            states.append(sim.data.qpos[:27].copy())
    rows = np.array(rows)
    tail = rows[-150:]
    quality = np.minimum(np.clip(rows[:, 0] / .94, 0, 1), np.clip(rows[:, 1] / 1.5, 0, 1)) * np.maximum(0, np.minimum(rows[:, 2], rows[:, 3]))
    score = 5 * np.mean(quality[-150:]) + 2 * np.max(quality) + np.mean(quality) - .05 * np.mean(np.minimum(tail[:, 4] + tail[:, 5], 10))
    return score, rows, qs, states


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iterations', type=int, default=0)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--output', default='artifacts/manual_getup')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sim = HumanoidSim(physics_dt=.001, action_mapping='joint_limits')
    sim.model.geom_contype[sim._robot_geoms] = 1
    # Parked projectiles never participate in the forward experiments.
    for ball in sim._balls:
        sim.model.geom_contype[ball.geom_id] = 0
        sim.model.geom_conaffinity[ball.geom_id] = 0
    rng = np.random.default_rng(20261103)
    mean = np.r_[DEFAULT.ravel(), [1., 1., 1.5, 2.]]
    low = np.tile([-.5, -1.6, .0, -.8, -2.4, .0, .08], 4)
    high = np.tile([.8, 1.1, 2.3, .8, 1.2, 2.3, .65], 4)
    low = np.r_[low, [.4] * 4]
    high = np.r_[high, [3.] * 4]
    std = np.r_[np.tile([.22, .4, .35, .25, .55, .4, .12], 4), [.4]*4]
    best = mean.copy()
    score, rows, _, _ = trajectory(sim, mean[:28].reshape(4,7), mean[28:])
    print('initial', score, 'tail', rows[-150:].mean(0), flush=True)
    best_score = score
    begin = time.monotonic()
    resume = out / 'trajectory_parameters.npz'
    if resume.exists():
        d = np.load(resume)
        mean = best = d['parameters'].copy()
        best_score = trajectory(sim, mean[:28].reshape(4,7), mean[28:])[0]
    for it in range(args.iterations):
        candidates = np.clip(mean + rng.normal(size=(args.population, len(mean))) * std, low, high)
        candidates[0] = best
        scores = np.array([trajectory(sim, c[:28].reshape(4,7), c[28:])[0] for c in candidates])
        order = np.argsort(scores)[::-1]
        elite = candidates[order[:max(4, args.population//6)]]
        mean = .2 * mean + .8 * elite.mean(0)
        std = np.maximum(.02, .3 * std + .7 * elite.std(0))
        if scores[order[0]] > best_score:
            best_score = float(scores[order[0]])
            best = candidates[order[0]].copy()
        np.savez(resume, parameters=best, score=best_score)
        print(json.dumps({'iteration': it, 'best': best_score, 'elapsed_s': round(time.monotonic()-begin,1)}), flush=True)
    score, rows, qs, states = trajectory(sim, best[:28].reshape(4,7), best[28:], keep=True)
    np.savez(resume, parameters=best, score=score, metrics=rows, targets=np.array(qs), qpos=np.array(states))
    print('best', score, best, 'tail', rows[-150:].mean(0), flush=True)


if __name__ == '__main__':
    main()
