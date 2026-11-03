"""Forward-only edits encoding a neutral standing region inside existing ELUs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from robot_rl.checkpoint import InferencePolicy
from robot_rl.env import HumanoidBalanceEnv
from tools.inspect_getup_policy import replay
import tools.tune_getup_forward as audit

BASE = None
PROBES = None


def configure(path, probes):
    global BASE, PROBES
    torch.set_grad_enabled(False)
    audit.POLICY = InferencePolicy(path)
    audit.POLICY.network.requires_grad_(False)
    BASE = {k: v.clone() for k, v in audit.POLICY.network.state_dict().items()}
    PROBES = np.load(probes)


def edit(parameters):
    threshold, sharpness, suppression, gate_strength, first, second = parameters
    first, second = int(first), int(second)
    s = {k: v.clone() for k, v in BASE.items()}
    # Reuse two low-influence neurons. Height is existing physical input 0.
    s['2.bias'] += s['2.weight'][:, first] * float(PROBES['mean_first'][first])
    s['0.weight'][first].zero_()
    s['0.weight'][first, 0] = sharpness
    s['0.bias'][first] = -sharpness * threshold
    s['2.weight'][:, first] = -suppression
    s['2.bias'] -= suppression
    s['4.bias'] += s['4.weight'][:, second] * float(PROBES['mean_second'][second])
    s['2.weight'][second].zero_()
    s['2.weight'][second, first] = -gate_strength
    s['2.bias'][second] = -gate_strength
    s['4.weight'][:, second] = 0.
    # At sufficient height all original final ELUs approach -1, and this
    # cancellation neuron also approaches -1: resulting action approaches zero.
    s['4.weight'][:, second] = s['4.bias'] - s['4.weight'].sum(dim=1)
    audit.POLICY.network.load_state_dict(s)
    if 'observations' in PROBES:
        hidden = audit.POLICY.network[:4](torch.as_tensor(PROBES['observations'])).numpy()
        x = hidden + 1.
        target = PROBES['targets']
        current = x @ s['4.weight'].numpy().T
        regularization = .0001 * np.trace(x @ x.T) / len(x)
        correction = np.linalg.solve(x.T @ x + regularization * np.eye(x.shape[1]), x.T @ (target - current))
        s['4.weight'] += torch.as_tensor(correction.T, dtype=torch.float32)
        s['4.bias'] = s['4.weight'].sum(dim=1)
        audit.POLICY.network.load_state_dict(s)
    return s


def evaluate(task):
    index, parameters = task
    edit(parameters)
    results = []
    for pose in ('stand', 'supine', 'prone'):
        result, _ = audit.run_pose(pose)
        results.append(result)
        if not result['success']:
            break
    valid = len(results) == 3 and all(r['success'] for r in results)
    natural = valid and all(r['tail']['arm_angle_mse'] < .05 and r['tail']['upright'] > .98
                            and r['tail']['speed'] < .05 and r['tail']['spin'] < .1 for r in results)
    cost = sum(r['cost'] for r in results) + (0. if valid else 1000.)
    return dict(index=index, parameters=list(parameters), valid=valid, natural=natural, cost=cost, poses=results)


def export(path, destination, parameters, probes, manifest):
    configure(path, probes)
    state = edit(parameters)
    with zipfile.ZipFile(path) as source:
        weights = torch.load(io.BytesIO(source.read('policy.pth')), map_location='cpu', weights_only=True)
        for layer, prefix in ((0, 'mlp_extractor.policy_net.0'), (2, 'mlp_extractor.policy_net.2'), (4, 'action_net')):
            for field in ('weight', 'bias'):
                weights[f'{prefix}.{field}'] = state[f'{layer}.{field}'].clone()
        data = io.BytesIO()
        torch.save(weights, data)
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                if item.filename == 'forward_edit.json':
                    continue
                payload = source.read(item.filename)
                if item.filename == 'policy.pth':
                    payload = data.getvalue()
                if item.filename == 'policy.optimizer.pth':
                    optimizer = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
                    optimizer['state'] = {}
                    buf = io.BytesIO(); torch.save(optimizer, buf); payload = buf.getvalue()
                target.writestr(item, payload)
            target.writestr('forward_edit.json', json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    source = out / 'original.zip'; source.write_bytes(Path(args.policy).read_bytes())
    torch.set_grad_enabled(False)
    p = InferencePolicy(source); p.network.requires_grad_(False)
    firsts, seconds, observations, targets = [], [], [], []
    for pose in ('stand', 'supine', 'prone'):
        env = HumanoidBalanceEnv(curriculum='getup')
        obs, _ = env.reset(options={'pose': pose})
        for step in range(1000):
            if step % 10 == 0:
                firsts.append(p.network[:2](torch.as_tensor(obs)).numpy())
                seconds.append(p.network[:4](torch.as_tensor(obs)).numpy())
                observations.append(obs.copy())
                target = p.network(torch.as_tensor(obs)).numpy()
                if step > 800 or pose == 'stand':
                    target = np.zeros_like(target)
                targets.append(target)
            obs, *_ = env.step(p.predict(obs)[0])
        env.close()
    env = HumanoidBalanceEnv(curriculum='getup')
    obs, _ = env.reset(options={'pose': 'stand'})
    for step in range(200):
        if step % 5 == 0:
            observations.append(obs.copy()); targets.append(np.zeros(20))
        obs, *_ = env.step(np.zeros(20))
    env.close()
    mean_first, mean_second = np.mean(firsts, axis=0), np.mean(seconds, axis=0)
    np.savez(out / 'probes.npz', mean_first=mean_first, mean_second=mean_second,
             observations=np.asarray(observations), targets=np.asarray(targets))
    first_rank = np.argsort(np.std(firsts, axis=0) * p.network[2].weight.norm(dim=0).numpy())[:3]
    second_rank = np.argsort(np.std(seconds, axis=0) * p.network[4].weight.norm(dim=0).numpy())[:3]
    print('reused neurons', first_rank.tolist(), second_rank.tolist(), flush=True)
    tasks = []; index = 0
    for first, second in zip(first_rank, second_rank):
        for threshold in (.78, .82, .86, .89):
            for suppression in (5., 15., 40.):
                for gate in (2., 10.):
                    tasks.append((index, (threshold, 80., suppression, gate, int(first), int(second))))
                    index += 1
    history, best = [], None
    with ProcessPoolExecutor(max_workers=6, initializer=configure,
                             initargs=(str(source), str(out / 'probes.npz'))) as executor:
        for result in executor.map(evaluate, tasks):
            history.append(result)
            if result['valid'] and (best is None or (result['natural'], -result['cost']) > (best['natural'], -best['cost'])):
                best = result
                print(json.dumps({'best': best}), flush=True)
                export(source, out / 'getup.zip', best['parameters'], out / 'probes.npz', best)
            (out / 'search.json').write_text(json.dumps(dict(best=best, candidates=history), indent=2), encoding='utf-8')
            if len(history) % 12 == 0:
                print('evaluated', len(history), 'of', len(tasks), flush=True)
    print('complete', json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
