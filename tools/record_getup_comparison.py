"""Side-by-side getup videos for several fixed-weight policies from identical starts."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot_rl.checkpoint import InferencePolicy
from robot_rl.env import HumanoidBalanceEnv
from robot_rl.recovery import is_recovered

MODELS = [
    ('原模型 197M', 'runs/checkpoints/humanoid_ppo_getup_197120000_steps.zip'),
    ('从零构造 getup_manual', 'artifacts/manual_getup/getup_manual.zip'),
    ('前向微调 human-stand', 'runs/diagnostics/getup-human-stand-20261103/getup.zip'),
]
STARTS = [('仰躺', 'supine', 10000), ('趴地', 'prone', 10000), ('随机', 'random', 10003)]
FONT = None


def font(size):
    for name in ('msyh.ttc', 'simhei.ttf', 'simsun.ttc'):
        try:
            return ImageFont.truetype(f'C:/Windows/Fonts/{name}', size)
        except OSError:
            pass
    return ImageFont.load_default()


def rollout(path, pose, seed, steps):
    policy = InferencePolicy(path)
    env = HumanoidBalanceEnv(curriculum='getup', max_episode_steps=steps, render_mode='rgb_array')
    obs, info = env.reset(seed=seed, options={'pose': pose})
    initial = env.sim.data.qpos.copy()
    frames, labels = [env.render()], []
    run = 0
    first = None
    try:
        for step in range(steps):
            obs, _, terminated, truncated, info = env.step(policy.predict(obs)[0])
            run = run + 1 if is_recovered(info) else 0
            if run == 100 and first is None:
                first = (step - 99) * .02
            if step % 2 == 1:
                frames.append(env.render())
                labels.append(run >= 100)
            if terminated or truncated:
                break
        q = info['joint_pos']
        summary = dict(steps=step + 1, success=bool(run >= 250), stable_start_seconds=first,
                       base_height=float(info['base_height']), arm_mse=float(np.mean(q[14:] ** 2)),
                       speed=float(np.linalg.norm(info['base_lin_vel'])))
        return frames, labels, summary, initial
    finally:
        env.close()


def annotate(frame, title, stable, t):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 34), fill=(20, 20, 20))
    draw.text((10, 4), title, font=FONT, fill=(255, 255, 255))
    status = '已站稳' if stable else ''
    draw.text((image.width - 170, 4), f'{t:5.1f}s {status}', font=FONT,
              fill=(120, 255, 120) if stable else (255, 255, 255))
    return np.asarray(image)


def main():
    global FONT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/diagnostics/getup-compare-20261103')
    parser.add_argument('--seconds', type=float, default=20.)
    args = parser.parse_args()
    FONT = font(20)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    steps = int(round(args.seconds / .02))
    report = {}
    for start_name, pose, seed in STARTS:
        columns, summaries, initials = [], {}, []
        for model_name, path in MODELS:
            frames, stable, summary, initial = rollout(path, pose, seed, steps)
            initials.append(initial)
            summaries[model_name] = summary
            stable = [False] + stable
            frames += [frames[-1]] * (steps // 2 + 1 - len(frames))
            stable += [stable[-1]] * (len(frames) - len(stable))
            columns.append([annotate(f, model_name, s, i * .04) for i, (f, s) in enumerate(zip(frames, stable))])
            print(json.dumps({start_name: {model_name: summary}}, ensure_ascii=False), flush=True)
        same = max(float(np.abs(initials[0] - x).max()) for x in initials[1:])
        report[start_name] = dict(pose=pose, seed=seed, max_initial_qpos_difference=same, models=summaries)
        video = out / f'{pose}_seed{seed}.mp4'
        with imageio.get_writer(video, fps=25, codec='libx264', quality=7, macro_block_size=8) as writer:
            for parts in zip(*columns):
                writer.append_data(np.concatenate(parts, axis=1))
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
