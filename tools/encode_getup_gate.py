"""Encode a physical 'stand still in the neutral posture' gate into actor weights.

Forward simulation only: no backward(), optimizer, learn() or autograd.
Gate score uses physical observations only: pelvis height, upper-arm height
above the pelvis and pelvis vertical velocity.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np
import torch

from robot_rl.checkpoint import InferencePolicy
from robot_rl.env import HumanoidBalanceEnv
from robot_rl.recovery import is_recovered

HEIGHT, VZ, ARMS = 0, 7, (26, 52)
ARM_UP = .678
STEEP = 20000.
POLICY = None
MODE = None


def score_vector(a, b, u=0., p=0.):
    w = np.zeros(384, np.float32)
    w[HEIGHT] = 1.
    w[list(ARMS)] = a
    w[VZ] = b
    w[1] = u  # heading-local pelvis quaternion w: 1 when upright
    w[[174, 175]] = p  # foot pressure / body weight: about 0.5 each when standing
    return w, -a * ARM_UP - u - p


def external_gate(obs, gate):
    a, b, low, high, *u = gate
    w, c = score_vector(a, b, *u)
    s = float(obs @ w + c)
    return low < s < high


def init(path, mode):
    global POLICY, MODE
    torch.set_grad_enabled(False)
    POLICY = InferencePolicy(path)
    MODE = mode


def rollout(task):
    gate, pose, seed, perturb = task
    env = HumanoidBalanceEnv(curriculum='getup', max_episode_steps=1000)
    obs, info = env.reset(seed=seed, options={'pose': pose})
    s = env.sim
    if perturb:
        rng = np.random.default_rng(seed)
        s.data.qpos[s.qpos_adr] = np.clip(s.data.qpos[s.qpos_adr] + rng.uniform(-perturb, perturb, s.n_act),
                                          s.q_low, s.q_high)
        mujoco.mj_forward(s.model, s.data)
        s._plant_on_ground()
        env._history.reset(s)
        obs = env._observation()
        info = env._info(0., s.foot_contact)
        info['feet_now'] = s.foot_contact.copy()
        env._reward.prime(info)
    feet_ids = [mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ('right_foot', 'left_foot')]
    run = longest = 0
    first = None
    flips = 0
    previous = False
    tail = []
    inverted = 0.
    for step in range(1000):
        if MODE == 'external':
            on = external_gate(obs, gate)
            action = np.zeros(20, np.float32) if on else POLICY.predict(obs)[0]
        else:
            action = POLICY.predict(obs)[0]
            on = bool(np.abs(action).max() < 1e-3)
        flips += int(on != previous)
        previous = on
        obs, _, terminated, truncated, info = env.step(action)
        run = run + 1 if is_recovered(info) else 0
        longest = max(longest, run)
        if run == 100 and first is None:
            first = round((step - 99) * .02, 2)
        inverted += .02 * float(min(info['upright'], info['chest_upright']) < -.2)
        q = info['joint_pos']
        feet = s.data.xpos[feet_ids, :2]
        tail.append([info['base_height'], np.linalg.norm(info['base_lin_vel']), np.linalg.norm(info['base_ang_vel']),
                     float(np.mean(q[14:] ** 2)), float(np.mean(q[:14] ** 2)), float(np.linalg.norm(feet[0] - feet[1])),
                     float(np.abs(action).mean())])
        if terminated or truncated:
            break
    env.close()
    m = np.mean(tail[-200:], 0)
    return dict(gate=list(gate) if gate else None, pose=pose, seed=seed, perturb=perturb,
                success=bool(run >= 250 and step == 999), first_stable=first, longest=round(longest * .02, 2),
                flips=flips, inverted=round(inverted, 2), height=round(m[0], 3), speed=round(m[1], 3),
                spin=round(m[2], 3), arm_mse=round(m[3], 4), leg_mse=round(m[4], 4), feet_gap=round(m[5], 3),
                mean_abs_action=round(m[6], 4))


STARTS = ([('supine', 10000, 0.), ('prone', 10000, 0.), ('stand', 10000, 0.)]
          + [('random', s, 0.) for s in range(10000, 10030)]
          + [(p, s, .03) for p in ('supine', 'prone') for s in (1, 2, 3)])


def influence(policy):
    """Forward probes: activation spread times outgoing weight norm."""
    net = policy.network
    acts1, acts2 = [], []
    env = HumanoidBalanceEnv(curriculum='getup', max_episode_steps=1000)
    probes = [(p, 10000) for p in ('supine', 'prone', 'stand')] + [('random', s) for s in range(20000, 20012)]
    probes += [(p, s) for p in ('supine', 'prone') for s in range(20000, 20004)]
    for pose, seed in probes:
        obs, _ = env.reset(seed=seed, options={'pose': pose})
        for _ in range(500):
            x = torch.as_tensor(obs)
            h1 = net[1](net[0](x))
            acts1.append(h1.numpy())
            acts2.append(net[3](net[2](h1)).numpy())
            obs, *_ = env.step(policy.predict(obs)[0])
    env.close()
    a1, a2 = np.asarray(acts1), np.asarray(acts2)
    w2, w3 = net[2].weight.numpy(), net[4].weight.numpy()
    i1 = a1.std(0) * np.linalg.norm(w2, axis=0)
    i2 = a2.std(0) * np.linalg.norm(w3, axis=0)
    return a1, a2, i1, i2


def reconstruction_error(acts):
    """Residual std when each unit is predicted linearly from all other units."""
    x = np.c_[acts, np.ones(len(acts))]
    precision = np.linalg.inv(x.T @ x / len(x) + 1e-3 * np.eye(x.shape[1]))
    return 1. / np.sqrt(np.diag(precision)[:-1])


def _fit(acts, keep, targets):
    x = np.c_[acts[:, keep], np.ones(len(acts))]
    return np.linalg.solve(x.T @ x + 1e-3 * np.eye(x.shape[1]), x.T @ targets)


def select_units(a1, a2, w2, b2, w3, b3, count=2, skip=0):
    """Greedy forward evaluation: reuse units that least change probe actions."""
    elu = lambda z: np.where(z > 0, z, np.expm1(np.minimum(z, 0)))
    squash = lambda y: np.clip(2. * np.tanh(y / 2.), -1., 1.)
    reference = squash(a2 @ w3.T + b3)
    rng = np.random.default_rng(0)
    rows = rng.choice(len(a1), min(4000, len(a1)), replace=False)
    h1, ref = a1[rows], reference[rows]
    chosen = []
    candidates = np.argsort(reconstruction_error(a1) * np.linalg.norm(w2, axis=0))[:40]
    for _ in range(count):
        scores = []
        for j in candidates:
            if j in chosen:
                scores.append(np.inf)
                continue
            group = chosen + [int(j)]
            keep = np.setdiff1d(np.arange(a1.shape[1]), group)
            coef = _fit(a1, keep, a1[:, group])
            approx = h1.copy()
            approx[:, group] = np.c_[h1[:, keep], np.ones(len(h1))] @ coef
            y = squash(elu(approx @ w2.T + b2) @ w3.T + b3)
            scores.append(np.abs(y - ref).mean())
        chosen.append(int(candidates[int(np.argsort(scores)[skip if not chosen else 0])]))
    second_candidates = np.argsort(reconstruction_error(a2) * np.linalg.norm(w3, axis=0))[:30]
    scores = []
    for k in second_candidates:
        keep = np.setdiff1d(np.arange(a2.shape[1]), [k])
        coef = _fit(a2, keep, a2[:, [k]])
        approx = a2[rows].copy()
        approx[:, [k]] = np.c_[approx[:, keep], np.ones(len(rows))] @ coef
        scores.append(np.abs(squash(approx @ w3.T + b3) - ref).mean())
    second = int(second_candidates[int(np.argsort(scores)[skip])])
    return np.asarray(chosen), second, float(np.min(scores))


def encode(source, destination, gate, rank1=0, rank2=0, probes=None, bump=True):
    torch.set_grad_enabled(False)
    policy = InferencePolicy(source)
    a1, a2, i1, i2 = probes or influence(policy)
    net = policy.network
    w1, b1 = net[0].weight.numpy().copy(), net[0].bias.numpy().copy()
    w2, b2 = net[2].weight.numpy().copy(), net[2].bias.numpy().copy()
    w3, b3 = net[4].weight.numpy().copy(), net[4].bias.numpy().copy()
    first, second, _ = select_units(a1, a2, w2, b2, w3, b3, count=4 if bump else 2, skip=rank1)
    # Re-express the reused units from the remaining units by least squares on
    # forward-probe activations (closed form, no gradients).
    keep1 = np.setdiff1d(np.arange(a1.shape[1]), first)
    x1 = np.c_[a1[:, keep1], np.ones(len(a1))]
    coef1 = np.linalg.lstsq(x1.T @ x1 + 1e-3 * np.eye(x1.shape[1]), x1.T @ a1[:, first], rcond=None)[0]
    w2[:, keep1] += w2[:, first] @ coef1[:-1].T
    b2 += w2[:, first] @ coef1[-1]
    keep2 = np.setdiff1d(np.arange(a2.shape[1]), [second])
    x2 = np.c_[a2[:, keep2], np.ones(len(a2))]
    coef2 = np.linalg.lstsq(x2.T @ x2 + 1e-3 * np.eye(x2.shape[1]), x2.T @ a2[:, second], rcond=None)[0]
    w3[:, keep2] += np.outer(w3[:, second], coef2[:-1])
    b3 += w3[:, second] * coef2[-1]
    a, b, low, high, *u = gate
    w, c = score_vector(a, b, *u)
    # Gate = (u0-u1) - (u2-u3): clipped steps at low and high, a bump in [0, 1].
    for unit, threshold, offset in zip(first, (low, low, high, high)[:len(first)], (0., -1., 0., -1.)):
        w1[unit] = STEEP * w
        b1[unit] = STEEP * (c - threshold) + offset
    signs = np.array([1., -1., -1., 1.][:len(first)], np.float32)
    big = 300.
    # On: every hidden-2 pre-activation gets -300, so ELU saturates at -1.
    w2[:, first] = -big * signs
    level = 1.
    # Dedicated hidden-2 unit: -1 (saturated) when off, +level when on.
    w2[second] = 0.
    w2[second, first] = (big + level) * signs
    b2[second] = -big
    others = np.ones(w3.shape[1], bool)
    others[second] = False
    # Off: dedicated unit is -1 and contributes -w_k, cancelled by b3 += w_k.
    # On: others are -1, dedicated is +level, so y = b3 + w_k(1+level) - total = 0.
    total = w3[:, others].sum(1)
    w_k = (total - b3) / (level + 1.)
    w3[:, second] = w_k
    b3 = b3 + w_k
    state = {'0.weight': w1, '0.bias': b1, '2.weight': w2, '2.bias': b2, '4.weight': w3, '4.bias': b3}
    source_bytes = Path(source).read_bytes()
    with zipfile.ZipFile(source) as archive:
        weights = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
        for number, prefix in ((0, 'mlp_extractor.policy_net.0'), (2, 'mlp_extractor.policy_net.2'), (4, 'action_net')):
            for field in ('weight', 'bias'):
                weights[f'{prefix}.{field}'] = torch.as_tensor(state[f'{number}.{field}'])
        payload = io.BytesIO()
        torch.save(weights, payload)
        optimizer = torch.load(io.BytesIO(archive.read('policy.optimizer.pth')), map_location='cpu', weights_only=True)
        optimizer['state'] = {}
        opt_payload = io.BytesIO()
        torch.save(optimizer, opt_payload)
        manifest = dict(method='forward-only analytic gate encoding',
                        gate=dict(arm_coeff=a, vz_coeff=b, upright_coeff=(u or [0.])[0], pressure_coeff=(u[1:] or [0.])[0], low=low, high=high, steepness=STEEP),
                        first_layer_units=first.tolist(), second_layer_unit=second,
                        source_sha256=hashlib.sha256(source_bytes).hexdigest(), gradient_enabled=False)
        replaced = {'policy.pth': payload.getvalue(), 'policy.optimizer.pth': opt_payload.getvalue()}
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as out:
            for item in archive.infolist():
                if item.filename == 'forward_edit.json':
                    continue
                out.writestr(item, replaced.get(item.filename, None) or archive.read(item.filename))
            out.writestr('forward_edit.json', json.dumps(manifest, indent=2))
    return manifest


def summarize(rows):
    ok = [r for r in rows if r['success']]
    return dict(success=f"{len(ok)}/{len(rows)}",
                arm_mse=round(float(np.mean([r['arm_mse'] for r in ok])), 4) if ok else None,
                leg_mse=round(float(np.mean([r['leg_mse'] for r in ok])), 4) if ok else None,
                feet_gap=round(float(np.mean([r['feet_gap'] for r in ok])), 3) if ok else None,
                failed=[(r['pose'], r['seed'], r['perturb']) for r in rows if not r['success']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--mode', choices=('external', 'encoded', 'encode'), required=True)
    parser.add_argument('--gates', default='[[0,0,0.86,1.05]]')
    parser.add_argument('--output')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    gates = [tuple(g) for g in json.loads(args.gates)]
    if args.mode == 'encode':
        print(json.dumps(encode(args.policy, args.output, gates[0])))
        return
    tasks = [(g if args.mode == 'external' else None, *start) for g in gates for start in STARTS]
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(args.policy, args.mode)) as ex:
        rows = list(ex.map(rollout, tasks))
    for g in (gates if args.mode == 'external' else [None]):
        sub = [r for r in rows if (r['gate'] == list(g) if g else True)]
        print(json.dumps(dict(gate=g, **summarize(sub))), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
