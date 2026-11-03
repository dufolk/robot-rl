"""Derivative-free actor edits, evaluated by fixed-weight MuJoCo forward rollouts.

No learn(), backward(), optimizer, autograd or changes to the observation contract.
Only successful, objectively improved candidates replace the best candidate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from robot_rl.checkpoint import InferencePolicy
from robot_rl.env import HumanoidBalanceEnv
from robot_rl.recovery import is_recovered

POLICY = None
BASE = None
DIRECTIONS = None


def configure(path, direction_path):
    global POLICY, BASE, DIRECTIONS
    torch.set_grad_enabled(False)
    POLICY = InferencePolicy(path)
    POLICY.network.requires_grad_(False)
    BASE = {key: value.clone() for key, value in POLICY.network.state_dict().items()}
    DIRECTIONS = np.load(direction_path)


def apply(parameters):
    # Each hidden dense layer is edited through a scalar weight/bias gain.
    # The output edit is a state-sensitive, low-rank correction from forward probes.
    state = {key: value.clone() for key, value in BASE.items()}
    state['0.weight'] *= 1. + parameters[0]
    state['0.bias'] *= 1. + parameters[0]
    state['2.weight'] *= 1. + parameters[1]
    state['2.bias'] *= 1. + parameters[1]
    state['4.weight'] += torch.as_tensor(parameters[2] * DIRECTIONS['posture_w'] + parameters[3] * DIRECTIONS['arms_w'])
    state['4.bias'] += torch.as_tensor(parameters[2] * DIRECTIONS['posture_b'] + parameters[3] * DIRECTIONS['arms_b'])
    state['4.weight'] *= 1. + parameters[4]
    state['4.bias'] *= 1. + parameters[4]
    POLICY.network.load_state_dict(state)


def run_pose(pose, seed=10000, capture=False, perturb=0.):
    env = HumanoidBalanceEnv(curriculum='getup', max_episode_steps=1000)
    # Capture features for direct output-layer corrections using forward data only.
    features, logits, standing = [], [], []
    longest = consecutive = 0
    first_stable = None
    inverted = head_contact = motion = action_change = 0.
    tail = []
    previous = np.zeros(20)
    try:
        obs, info = env.reset(seed=seed, options={'pose': pose})
        if perturb:
            import mujoco
            rng = np.random.default_rng(seed)
            env.sim.data.qpos[env.sim.qpos_adr] += rng.uniform(-perturb, perturb, env.sim.n_act)
            env.sim.data.qpos[env.sim.qpos_adr] = np.clip(env.sim.data.qpos[env.sim.qpos_adr],
                                                        env.sim.q_low, env.sim.q_high)
            mujoco.mj_forward(env.sim.model, env.sim.data)
            env.sim._plant_on_ground()
            env._history.reset(env.sim)
            obs = env._observation()
            info = env._info(0., env.sim.foot_contact)
            info['feet_now'] = env.sim.foot_contact.copy()
            env._reward.prime(info)
        for step in range(1000):
            if capture and step % 10 == 0:
                hidden = POLICY.network[:-1](torch.as_tensor(obs)).numpy()
                features.append(hidden)
                logits.append(POLICY.network[-1](torch.as_tensor(hidden)).numpy())
                standing.append(is_recovered(info))
            action = POLICY.predict(obs)[0]
            obs, _, terminated, truncated, info = env.step(action)
            stable = is_recovered(info)
            consecutive = consecutive + 1 if stable else 0
            longest = max(longest, consecutive)
            if consecutive == 100 and first_stable is None:
                first_stable = (step + 1 - 100) * .02
            up = min(info['upright'], info['chest_upright'])
            inverted += float(up < -.2) * .02
            head_contact += float(info['floor_n'][-1] > 10.) * .02
            speed, spin = np.linalg.norm(info['base_lin_vel']), np.linalg.norm(info['base_ang_vel'])
            motion += .02 * (min(spin * spin, 100.) + min(speed * speed, 25.))
            action_change += float(np.mean((action - previous) ** 2))
            previous = action
            q = info['joint_pos']
            tail.append([up, speed, spin, float(np.mean(q[14:] ** 2)), float(np.mean(q[:14] ** 2)),
                         info['base_height'], info['head_height']])
            if terminated or truncated:
                break
        tail_mean = np.mean(tail[-200:], axis=0)
        success = bool(info.get('is_success', False)) and stable and longest >= 250
        cost = ((first_stable if first_stable is not None else 30.)
                + 2. * inverted + .75 * head_contact + .08 * motion
                + .1 * action_change + 8. * tail_mean[3] + 3. * tail_mean[4]
                + 15. * (1. - tail_mean[0]) + 5. * tail_mean[1] + 2. * tail_mean[2])
        result = dict(pose=pose, seed=seed, perturb_radians=perturb, success=success, cost=float(cost),
                      stable_start_seconds=first_stable, longest_stable_seconds=longest * .02,
                      inverted_seconds=inverted, head_contact_seconds=head_contact, motion_integral=motion,
                      tail=dict(zip(('upright', 'speed', 'spin', 'arm_angle_mse', 'leg_waist_angle_mse',
                                     'base_height', 'head_height'), tail_mean.tolist())))
        return result, (features, logits, standing)
    finally:
        env.close()


def evaluate(task):
    index, parameters = task
    apply(parameters)
    results = []
    for pose in ('prone', 'supine', 'stand'):
        result, _ = run_pose(pose)
        results.append(result)
        if not result['success']:
            break
    valid = len(results) == 3 and all(r['success'] for r in results)
    cost = sum(r['cost'] for r in results) if valid else 1000. + sum(r['cost'] for r in results)
    return dict(index=index, parameters=np.asarray(parameters).tolist(), valid=valid, cost=cost, poses=results)


def make_directions(path, output):
    global POLICY, BASE
    torch.set_grad_enabled(False)
    POLICY = InferencePolicy(path)
    POLICY.network.requires_grad_(False)
    probes, raw, labels = [], [], []
    baseline = []
    for pose in ('stand', 'supine', 'prone'):
        result, (features, logits, standing) = run_pose(pose, capture=True)
        probes.extend(features)
        raw.extend(logits)
        labels.extend(standing)
        baseline.append(result)
    x = np.c_[np.asarray(probes), np.ones(len(probes))]
    y = np.asarray(raw)
    labels = np.asarray(labels)
    # Forward probes select a minimal-norm correction: preserve falling-state
    # logits, shrink standing logits toward the existing neutral joint targets.
    residual = -y * labels[:, None]
    gram = x @ x.T
    regularization = .002 * np.trace(gram) / len(gram)
    correction = x.T @ np.linalg.solve(gram + np.eye(len(gram)) * regularization, residual)
    posture, arms = correction.copy(), correction.copy()
    posture[:, 14:] = 0.
    arms[:, :14] = 0.
    np.savez(output / 'directions.npz', posture_w=posture[:-1].T.astype(np.float32),
             posture_b=posture[-1].astype(np.float32), arms_w=arms[:-1].T.astype(np.float32),
             arms_b=arms[-1].astype(np.float32))
    return baseline


def export(source, destination, parameters, directions, manifest):
    configure(source, directions)
    apply(parameters)
    modified = POLICY.network.state_dict()
    with zipfile.ZipFile(source) as archive:
        weights = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
        for number, prefix in ((0, 'mlp_extractor.policy_net.0'), (2, 'mlp_extractor.policy_net.2'), (4, 'action_net')):
            for field in ('weight', 'bias'):
                weights[f'{prefix}.{field}'] = modified[f'{number}.{field}'].clone()
        payload = io.BytesIO()
        torch.save(weights, payload)
        # A changed actor invalidates any saved Adam state: reset it if resumed.
        optimizer_payload = None
        if 'policy.optimizer.pth' in archive.namelist():
            optimizer = torch.load(io.BytesIO(archive.read('policy.optimizer.pth')),
                                   map_location='cpu', weights_only=True)
            optimizer['state'] = {}
            optimizer_payload = io.BytesIO()
            torch.save(optimizer, optimizer_payload)
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as target:
            for item in archive.infolist():
                if item.filename == 'policy.optimizer.pth':
                    target.writestr(item, optimizer_payload.getvalue())
                    continue
                target.writestr(item, payload.getvalue() if item.filename == 'policy.pth' else archive.read(item.filename))
            target.writestr('forward_edit.json', json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_bytes = Path(args.policy).read_bytes()
    source = output / 'original.zip'
    source.write_bytes(source_bytes)
    baseline = make_directions(source, output)
    print(json.dumps({'baseline': baseline}), flush=True)
    configure(source, output / 'directions.npz')
    best = evaluate((0, np.zeros(5)))
    history = [best]
    rng = np.random.default_rng(20261103)
    count = 1
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure,
                             initargs=(str(source), str(output / 'directions.npz'))) as executor:
        for generation in range(args.rounds):
            tasks = []
            if generation == 0:
                for posture, arms in ((0., .05), (0., .1), (0., .2), (.02, .05), (.05, .05), (.1, .1),
                                      (.2, .2), (.3, .3), (0., .4), (0., .6), (.01, .02), (.02, .02)):
                    tasks.append((count, np.array([.0001, -.0001, posture, arms, 0.])))
                    count += 1
            else:
                center = np.asarray(best['parameters'])
                scale = np.array([.0015, .0015, .025, .05, .003]) * (.75 ** ((generation - 1) // 3))
                for _ in range(6):
                    noise = rng.normal(size=5) * scale
                    for sign in (-1., 1.):
                        candidate = center + sign * noise
                        candidate[2:4] = np.clip(candidate[2:4], -.1, 1.)
                        tasks.append((count, candidate))
                        count += 1
            for result in executor.map(evaluate, tasks):
                history.append(result)
                if result['valid'] and result['cost'] < best['cost']:
                    best = result
                    print(json.dumps({'improvement': best, 'elapsed_seconds': time.monotonic() - started}), flush=True)
                    manifest = dict(method='forward-only derivative-free actor parameter search',
                                    source_sha256=hashlib.sha256(source_bytes).hexdigest(), best=best,
                                    baseline=baseline, gradient_enabled=False)
                    export(source, output / 'getup.zip', best['parameters'], output / 'directions.npz', manifest)
                (output / 'search.json').write_text(json.dumps(dict(best=best, candidates=history), indent=2), encoding='utf-8')
            print(json.dumps({'round': generation + 1, 'candidates': len(history), 'best_cost': best['cost'],
                              'elapsed_seconds': time.monotonic() - started}), flush=True)
    # Always deliver a checkpoint; record explicitly whether a valid edit won.
    manifest = dict(method='forward-only derivative-free actor parameter search',
                    source_sha256=hashlib.sha256(source_bytes).hexdigest(), best=best, baseline=baseline,
                    gradient_enabled=False, improved=best['index'] != 0)
    export(source, output / 'getup.zip', best['parameters'], output / 'directions.npz', manifest)
    print(json.dumps({'finished': str(output / 'getup.zip'), 'best': best}), flush=True)


if __name__ == '__main__':
    main()
