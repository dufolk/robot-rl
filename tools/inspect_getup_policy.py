"""Read-only checkpoint replay and explicit action-head scaling; never trains."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import torch

from robot_rl.checkpoint import InferencePolicy, read_metadata
from robot_rl.env import HumanoidBalanceEnv
from robot_rl.recovery import is_recovered


def write_png(path, image):
    image = np.asarray(image, dtype=np.uint8)
    height, width, _ = image.shape
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    rows = b''.join(b'\x00' + row.tobytes() for row in image)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def replay(policy, pose, seed, output, render=False):
    env = HumanoidBalanceEnv(curriculum='getup', max_episode_steps=1000,
                             render_mode='rgb_array' if render else None)
    rows, frames = [], []
    consecutive = longest = 0
    try:
        obs, info = env.reset(seed=seed, options={'pose': pose})
        if render:
            frames.append(env.render()[::2, ::2].copy())
        for step in range(1000):
            action = policy.predict(obs)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            stable = is_recovered(info)
            consecutive = consecutive + 1 if stable else 0
            longest = max(longest, consecutive)
            rows.append([.02 * (step + 1), info['base_height'], info['head_height'],
                         info['upright'], info['chest_upright'],
                         np.linalg.norm(info['base_lin_vel']), np.linalg.norm(info['base_ang_vel']),
                         np.mean(np.abs(action)), np.mean(np.abs(action) > .9), stable,
                         info.get('assist_n', env._assist_n)])
            if render and (step + 1) in (50, 100, 200, 400, 600, 800, 1000):
                frames.append(env.render()[::2, ::2].copy())
            if terminated or truncated:
                break
        values = np.asarray(rows)
        np.save(output.with_suffix('.npy'), values)
        if frames:
            while len(frames) < 8:
                frames.append(frames[-1])
            write_png(output.with_suffix('.png'), np.concatenate([
                np.concatenate(frames[:4], axis=1), np.concatenate(frames[4:8], axis=1)], axis=0))
        return {'pose': pose, 'seed': seed, 'steps': len(rows),
                'success': bool(info.get('is_success', False)), 'longest_stable_seconds': longest * .02,
                'tail_mean': dict(zip(('time', 'base_height', 'head_height', 'pelvis_up', 'chest_up',
                                      'speed', 'spin', 'mean_abs_action', 'action_over_0.9', 'stable', 'assist_n'),
                                     values[-200:].mean(axis=0).tolist()))}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.policy).resolve()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    payload = source.read_bytes()
    snapshot = output / source.name
    snapshot.write_bytes(payload)
    policy = InferencePolicy(snapshot)
    head = policy.network[-1]
    weight, bias = head.weight.detach().clone(), head.bias.detach().clone()
    result = {'source': str(source), 'sha256': hashlib.sha256(payload).hexdigest(),
              'metadata': read_metadata(snapshot, ('num_timesteps', 'getup_reward', 'humanup_stage', 'humanup_contract')),
              'observation_dim': policy.observation_space.shape[0],
              'frame_times_seconds': [0, 1, 2, 4, 8, 12, 16, 20], 'variants': []}
    # Named poses have deterministic resets; repeating seeds adds no coverage.
    variants = [('all', scale) for scale in (1., .5, .25, 0.)]
    variants += [('arms', scale) for scale in (.5, 0.)]
    for region, scale in variants:
        with torch.no_grad():
            head.weight.copy_(weight)
            head.bias.copy_(bias)
            affected = slice(None) if region == 'all' else slice(14, 20)
            head.weight[affected].mul_(scale)
            head.bias[affected].mul_(scale)
        for pose in ('stand', 'supine', 'prone'):
            for seed in (10000,):
                name = f'{region}-scale-{scale:g}_{pose}_{seed}'
                stats = replay(policy, pose, seed, output / name, render=scale in (1., .5))
                stats['head_scale'] = scale
                stats['affected_head_rows'] = region
                result['variants'].append(stats)
                print(json.dumps(stats), flush=True)
                (output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
