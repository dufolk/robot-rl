"""Find the fastest parallel training setup on this machine.

Each candidate (env count x worker count x device) runs real PPO
iterations, sampling plus update, for a short time and reports steady-state
steps/s. No checkpoints or logs are written. The fastest setup is printed as
a ready-to-run training command and saved to runs/parallel_tuning.json.

    .venv\\Scripts\\python.exe tools/tune_parallel.py            # standard search
    .venv\\Scripts\\python.exe tools/tune_parallel.py --quick    # fewer candidates
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def physical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except ImportError:
        return max(1, (os.cpu_count() or 8) // 2)


def gpu_name() -> str | None:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def candidates(quick: bool, logical: int, physical: int, has_gpu: bool):
    workers = sorted({max(1, w) for w in (physical // 2, physical, physical + (logical - physical) // 2, logical - 2, logical)})
    envs = (256, 512) if quick else (128, 256, 512, 1024)
    devices = ["cuda", "cpu"] if has_gpu and not quick else (["cuda"] if has_gpu else ["cpu"])
    for device in devices:
        for n_envs in envs:
            for n_workers in workers:
                if n_workers <= n_envs:
                    yield n_envs, n_workers, device


def measure(n_envs: int, n_workers: int, device: str, seconds: float) -> dict:
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from robot_rl.env import HumanoidBalanceEnv
    from robot_rl.humanup import initial_action_std
    from robot_rl.humanup_policy import RecoveryPolicy
    from robot_rl.humanup_ppo import DESIRED_KL, HumanUPPPO
    from robot_rl.vec import ShardedVecEnv

    torch.set_num_threads(1)
    env_kwargs = {"humanup_stage": "discovery", "reference_path": None}
    probe = HumanoidBalanceEnv(curriculum="getup", **env_kwargs)
    std = initial_action_std(probe.sim)
    probe.close()
    steps = 24
    vec = ShardedVecEnv(n_envs, curriculum="getup", seed=0, n_workers=n_workers, env_kwargs=env_kwargs, compact_infos=True)
    try:
        model = HumanUPPPO(RecoveryPolicy, vec, desired_kl=DESIRED_KL, learning_rate=2e-4, n_steps=steps,
                           batch_size=max(2, steps * n_envs // 4), n_epochs=5, target_kl=2 * DESIRED_KL,
                           ent_coef=.001, vf_coef=1., clip_range_vf=.2, max_grad_norm=1., device=device, verbose=0,
                           policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]},
                                          "activation_fn": torch.nn.ELU})
        with torch.no_grad():
            model.policy.log_std.copy_(torch.as_tensor(std, device=model.device).log())

        class Clock(BaseCallback):
            def __init__(self):
                super().__init__()
                self.marks = []

            def _on_rollout_start(self):
                self.marks.append((time.perf_counter(), self.model.num_timesteps))

            def _on_step(self):
                started = self.marks[1][0] if len(self.marks) > 1 else None
                return started is None or time.perf_counter() - started < seconds

        clock = Clock()
        model.learn(total_timesteps=10 ** 9, callback=clock, progress_bar=False)
        # Skip the first iteration (process start-up, CUDA warm-up).
        (t0, s0), (t1, s1) = clock.marks[1], (time.perf_counter(), model.num_timesteps)
        rate = (s1 - s0) / max(t1 - t0, 1e-9)
        return {"envs": n_envs, "workers": n_workers, "device": device, "steps_per_second": round(rate, 1),
                "iterations": len(clock.marks) - 1}
    finally:
        vec.close()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="只测 256/512 个环境和当前最可能的设备")
    parser.add_argument("--seconds", type=float, default=25., help="每个候选的稳态测量时长")
    parser.add_argument("--timesteps", type=int, default=300_000_000, help="写进推荐命令里的总步数")
    args = parser.parse_args()

    logical, physical = os.cpu_count() or 4, physical_cores()
    gpu = gpu_name()
    print(f"CPU {platform.processor() or platform.machine()}，物理核 {physical}，逻辑核 {logical}，GPU {gpu or '无'}")
    results = []
    plan = list(candidates(args.quick, logical, physical, gpu is not None))
    print(f"共 {len(plan)} 组候选，每组约 {args.seconds + 15:.0f} 秒。")
    for index, (n_envs, n_workers, device) in enumerate(plan, 1):
        try:
            result = measure(n_envs, n_workers, device, args.seconds)
        except Exception as exc:  # noqa: BLE001 - report and continue with other setups
            result = {"envs": n_envs, "workers": n_workers, "device": device, "error": repr(exc)}
        results.append(result)
        shown = f"{result['steps_per_second']:>8.0f} 步/秒" if "steps_per_second" in result else f"失败 {result['error']}"
        print(f"[{index:>2}/{len(plan)}] envs {n_envs:>4}  workers {n_workers:>3}  {device:<4}  {shown}", flush=True)

    ok = [r for r in results if "steps_per_second" in r]
    if not ok:
        raise SystemExit("所有候选都失败了，看上面的报错。")
    best = max(ok, key=lambda r: r["steps_per_second"])
    command = (f".venv\\Scripts\\python.exe -m robot_rl.train --curriculum getup --timesteps {args.timesteps} "
               f"--envs {best['envs']} --workers {best['workers']} --device {best['device']}")
    if os.name != "nt":
        command = command.replace(".venv\\Scripts\\python.exe", ".venv/bin/python")
    out = ROOT / "runs" / "parallel_tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cpu_logical": logical, "cpu_physical": physical, "gpu": gpu,
                               "results": results, "best": best, "command": command}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n最快：envs {best['envs']}，workers {best['workers']}，{best['device']}，约 {best['steps_per_second']:.0f} 步/秒。")
    print("注意：环境数越多 batch 越大，样本效率会变，推荐在最快和 256/512 之间权衡。")
    print(f"启动命令：\n  {command}")
    print(f"全部结果写入 {out}")


if __name__ == "__main__":
    main()
