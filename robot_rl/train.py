"""PPO 训练主循环。

用 `--curriculum` 选课程。getup-human 精修参考起身动作，impact 继续抗击打。
getup-human 和 impact 跑满预算，按固定协议评估并单独保存最佳模型。
DeepMimic 课程使用 1024/512 ReLU 网络和固定高斯探索标准差。
有 GPU 时更新放在 GPU 上。观测里没有炮弹，策略只能在被打中之后恢复，不能提前躲开。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections import deque
from pathlib import Path

# 每个环境进程只留一条计算线程。否则 numpy / torch 会按 24 个逻辑核再开线程，和仿真抢核。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import swanlab
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from swanlab.integration.sb3 import SwanLabCallback

from robot_rl.curriculum import CURRICULA, SUCCESS_RATE, SUCCESS_WINDOW, get_curriculum
from robot_rl.env import HumanoidBalanceEnv
from robot_rl.vec import N_ENVS, ShardedVecEnv, worker_count

RUN_DIR = Path("runs")
DASHBOARD_URL = "http://127.0.0.1:5092" 
# 检查点大约每十几万步存一次。只留最近几个，避免 runs/checkpoints 堆到几百个。
KEEP_CHECKPOINTS = 5
ENT_COEF = 1e-4
# 1e-4 时一轮 KL 只有 0.02。3e-4 时 KL 到 0.1、超过一半样本被裁剪，策略变差。
# 不设 KL 上限后，2e-4 跑满 4 个 epoch 会把 KL 推到 0.2 以上。
LEARNING_RATE = 1e-4
N_EPOCHS = 4
# SB3 在 approx_kl > 1.5 * target_kl 时停掉本轮还没做的 minibatch，已经做完的那一步留着。
# 0.05 的线是 0.075，学会之后的正常更新（KL 大约 0.07）会被掐掉。
# 0.5 的线是 0.75，给正常更新留余量，并限制严重偏离采样策略的后续更新。
TARGET_KL = 3.5
# 爬起做不到一万局全对。97% 已经是站稳的策略，再训会被一次大更新推开。
GETUP_SUCCESS_RATE = 0.97
BATCH_SIZE = 2048
LOG_STD_INIT = -1.5
# getup-human 从旧权重精修时，把探索拉回开训标准差。
GETUP_HUMAN_LOG_STD = LOG_STD_INIT
# 代码里的键保持英文。看板只在上报时换成中文。
METRIC_NAMES = {
    "curriculum/progress": "课程/进度",
    "rollout/ep_rew_mean": "采样/平均回报",
    "rollout/ep_len_mean": "采样/平均局长",
    "rollout/success_rate": "采样/成功率",
    "time/fps": "时间/每秒步数",
    "time/iterations": "时间/轮数",
    "time/time_elapsed": "时间/已用秒",
    "time/total_timesteps": "时间/总步数",
    "train/approx_kl": "训练/KL",
    "train/kl_from_init": "训练/相对初始KL",
    "train/clip_fraction": "训练/裁剪比例",
    "train/clip_range": "训练/裁剪范围",
    "train/entropy_loss": "训练/熵损失",
    "train/explained_variance": "训练/解释方差",
    "train/learning_rate": "训练/学习率",
    "train/loss": "训练/总损失",
    "train/n_updates": "训练/更新次数",
    "train/policy_gradient_loss": "训练/策略损失",
    "train/value_loss": "训练/价值损失",
    "train/std": "训练/动作标准差",
    "eval/recovery_success_rate": "评估/恢复成功率",
    "eval/motion_imitation": "评估/模仿",
    "eval/action_delta_rms": "评估/动作变化",
    "eval/motion_quality": "评估/动作质量",
    "recovery/holding_fraction": "恢复/保持比例",
    "motion/imitation": "动作/模仿",
    "motion/reference_pose": "动作/参考姿态",
    "motion/reference_velocity": "动作/参考速度",
    "motion/reference_ee": "动作/参考手脚",
    "motion/reference_com": "动作/参考质心",
    "reward/total": "奖励/合计",
    "reward/assist": "诊断/拉力",
    "reward/imitation": "奖励/模仿",
    "reward/dof_acc": "奖励/关节加速度",
    "reward/dof_vel": "奖励/关节速度",
    "reward/action_rate": "奖励/动作变化",
    "reward/torque": "奖励/力矩",
    "reward/dof_torque": "奖励/力矩平方",
    "reward/ang_vel": "奖励/角速度",
    "reward/ang_vel_xy": "奖励/横滚俯仰",
    "reward/lin_vel": "奖励/线速度",
    "reward/lin_vel_z": "奖励/竖直速度",
    "reward/feet_slide": "奖励/脚滑",
    "reward/energy": "奖励/能量",
    "reward/dof_limit": "奖励/关节限位",
    "reward/torque_limit": "奖励/力矩限位",
    "reward/flat": "奖励/躯干倾斜",
    "reward/undesired": "奖励/非脚着地",
    "reward/joint_pose": "奖励/关节姿态",
    "reward/termination": "奖励/终止",
    "reward/height": "奖励/高度",
    "reward/d_height": "奖励/高度增量",
    "reward/upright": "奖励/立起",
    "reward/feet": "奖励/双脚承重",
    "reward/d_feet": "奖励/脚力增量",
}


def metric_name(key: str) -> str:
    return METRIC_NAMES.get(key, key)


def _use_chinese_metric_names(logger) -> None:
    if getattr(logger, "_chinese_names", False):
        return
    original = logger.record

    def record(key, value, exclude=None):
        original(metric_name(key), value, exclude)

    logger.record = record
    logger._chinese_names = True


def success_threshold(curriculum: str) -> float | None:
    if curriculum in ("getup-human", "impact"):
        return None
    if curriculum == "getup-plus":
        return GETUP_SUCCESS_RATE
    return SUCCESS_RATE


def ready_to_stop(successes: list[bool] | deque[bool], window: int = SUCCESS_WINDOW, threshold: float = SUCCESS_RATE) -> bool:
    if len(successes) < window:
        return False
    recent = list(successes)[-window:]
    return (sum(recent) / window) > threshold


def train(
    timesteps: int,
    n_envs: int = N_ENVS,
    seed: int = 0,
    device: str = "auto",
    save_path: Path | None = None,
    n_steps: int | None = None,
    curriculum: str = "stand",
    load: Path | None = None,
) -> PPO:
    import torch

    torch.set_num_threads(1)
    _drop_unsupported_gpu_power()
    course = get_curriculum(curriculum)
    deepmimic = course.name in ("getup-human", "impact")
    prefix = f"humanoid_ppo_{course.name}" + ("_deepmimic_state" if deepmimic else "")
    save_path = RUN_DIR / prefix if save_path is None else Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps = 64 if n_steps is None else int(n_steps)
    batch_size = min(BATCH_SIZE, steps * n_envs)
    ent_coef = 0.0 if course.name == "stand" or deepmimic else ENT_COEF
    learning_rate = 5e-5 if course.name == "getup-human" else LEARNING_RATE
    # Validate before spawning hundreds of environments. Different observations
    # define a new policy; silently padding old input weights is not a resume.
    if load is not None:
        probe = HumanoidBalanceEnv(curriculum=course.name)
        try:
            loaded = PPO.load(str(load), device="cpu")
            if loaded.observation_space != probe.observation_space or loaded.action_space != probe.action_space:
                raise ValueError(
                    f"检查点空间 {loaded.observation_space.shape}/{loaded.action_space.shape} "
                    f"与本课程 {probe.observation_space.shape}/{probe.action_space.shape} 不兼容。"
                    "策略接收 189 维身体、接触和动作历史。旧 89/157/160/245 维模型不能续训；请从头训练，impact 可加载新版 getup-human。"
                )
            del loaded
        finally:
            probe.close()
    workers = 1 if n_envs == 1 else worker_count(n_envs)
    print(
        f"并行 {n_envs} 个人形，{workers} 个进程各自串行推动，batch {batch_size}，"
        f"单轮 KL 超过 {1.5 * TARGET_KL:g} 就停下，device {device}。"
    )
    if n_envs == 1:
        vec = make_vec_env(
            HumanoidBalanceEnv,
            n_envs=1,
            seed=seed,
            env_kwargs={"curriculum": course.name},
            vec_env_cls=DummyVecEnv,
            monitor_kwargs={"info_keywords": ("is_success",)},
        )
    else:
        vec = ShardedVecEnv(n_envs, curriculum=course.name, seed=seed, n_workers=workers)
    if load is None:
        model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=learning_rate,
            n_steps=steps,
            batch_size=batch_size,
            n_epochs=N_EPOCHS,
            target_kl=TARGET_KL,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            device=device,
            verbose=1,
            policy_kwargs={
                "net_arch": {"pi": [1024, 512] if deepmimic else [256, 256],
                             "vf": [1024, 512] if deepmimic else [256, 256]},
                "activation_fn": torch.nn.ReLU if deepmimic else torch.nn.Tanh,
                "log_std_init": LOG_STD_INIT,
            },
        )
        already = 0
    else:
        model = PPO.load(
            str(load),
            env=vec,
            device=device,
            learning_rate=learning_rate,
            n_steps=steps,
            batch_size=batch_size,
            n_epochs=N_EPOCHS,
            target_kl=TARGET_KL,
        )
        already = int(model.num_timesteps)
        print(f"从 {load} 续训，已有 {already} 步。熵系数按本课程设为 {ent_coef}。")
    model.ent_coef = ent_coef
    if deepmimic:
        model.policy.log_std.requires_grad_(False)
    model._stats_window_size = SUCCESS_WINDOW
    model.ep_info_buffer = deque(maxlen=SUCCESS_WINDOW)
    model.ep_success_buffer = deque(maxlen=SUCCESS_WINDOW)
    remaining = int(timesteps) - already
    if remaining <= 0:
        raise ValueError(f"检查点已有 {already} 步，不少于 --timesteps {int(timesteps)}")
    log_dir = save_path.parent / "swanlog"
    callbacks: list[BaseCallback] = [
        SwanLabCallback(
            project="humanoid-balance",
            experiment_name=f"ppo-{course.name}" + ("-deepmimic-state" if deepmimic else ""),
            description=course.description,
            log_dir=str(log_dir),
            mode="local",
        ),
        _RewardCurves(2 * course.horizon),
        _PolicyDriftFromInit(),
        _CurriculumProgress(),
        _StopOnSuccess(course.name, save_path.parent / "checkpoints" / f"{prefix}_best.zip"),
    ]
    if course.name in ("getup-human", "impact"):
        callbacks.append(_RecoveryEvaluation(course.name, save_path.parent / "checkpoints" / f"{prefix}_best.zip"))
    if int(timesteps) >= 10_000:
        checkpoint_dir = save_path.parent / "checkpoints"
        prune_checkpoints(checkpoint_dir, prefix=prefix)
        callbacks.append(
            _KeepRecentCheckpoints(
                save_freq=max(steps * 20, 10_000),
                save_path=str(checkpoint_dir),
                name_prefix=prefix,
            )
        )
    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=already == 0,
        )
        model.save(str(save_path))
    finally:
        swanlab.finish()
        vec.close()
    return model


def prune_checkpoints(directory: Path, prefix: str | None = None, keep: int = KEEP_CHECKPOINTS) -> int:
    """删掉旧检查点。`prefix` 为空时，每个课程各自保留最近 `keep` 个。"""
    if keep < 1:
        raise ValueError(f"keep 至少为 1，收到 {keep}")
    if not directory.is_dir():
        return 0
    groups = _checkpoint_groups(directory)
    if prefix is not None:
        groups = {prefix: groups.get(prefix, [])}
    removed = 0
    for items in groups.values():
        items.sort(key=lambda item: item[1].stat().st_mtime)
        for _, path in items[:-keep]:
            path.unlink()
            removed += 1
    if removed:
        print(f"检查点只保留每个课程最近 {keep} 个，删除了 {removed} 个旧模型。")
    return removed


def _checkpoint_groups(directory: Path) -> dict[str, list[tuple[int, Path]]]:
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in directory.glob("*_steps.zip"):
        stem = path.name[: -len(".zip")]
        head, _, tail = stem.rpartition("_")
        if tail != "steps":
            continue
        name, _, steps = head.rpartition("_")
        if not name or not steps.isdigit():
            continue
        groups.setdefault(name, []).append((int(steps), path))
    return groups


def _reopen_action_noise(model: PPO, log_std_init: float) -> tuple[float, float]:
    import torch

    log_std = model.policy.log_std
    before = float(log_std.detach().exp().mean())
    with torch.no_grad():
        log_std.fill_(float(log_std_init))
    slot = model.policy.optimizer.state.get(log_std)
    if slot is not None:
        for moment in slot.values():
            if torch.is_tensor(moment):
                moment.zero_()
    after = float(log_std.detach().exp().mean())
    return before, after


def mean_policy_kl(policy, reference, observations) -> float:
    """当前策略相对冻结副本的平均 KL(π‖π0)，在同一批观测上算。"""
    import torch
    from torch.distributions import kl_divergence

    obs = observations.reshape(-1, observations.shape[-1])
    obs_tensor, _ = policy.obs_to_tensor(obs)
    with torch.no_grad():
        current = policy.get_distribution(obs_tensor)
        baseline = reference.get_distribution(obs_tensor)
        kl = kl_divergence(current.distribution, baseline.distribution)
        return float(kl.mean().item())


class _PolicyDriftFromInit(BaseCallback):
    """初始化或 load 后冻结一份策略，每轮上报相对它的 KL。"""

    def __init__(self) -> None:
        super().__init__()
        self._reference = None

    def _on_training_start(self) -> None:
        # 排在动作标准差重置之后；此时还没有梯度更新。
        self._reference = copy.deepcopy(self.model.policy)
        self._reference.set_training_mode(False)
        for parameter in self._reference.parameters():
            parameter.requires_grad_(False)
        self.logger.record("train/kl_from_init", 0.0)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self._reference is None:
            return
        observations = self.model.rollout_buffer.observations
        kl = mean_policy_kl(self.model.policy, self._reference, observations)
        self.logger.record("train/kl_from_init", kl)


class _KeepRecentCheckpoints(CheckpointCallback):
    def _on_step(self) -> bool:
        training = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            prune_checkpoints(Path(self.save_path), self.name_prefix)
        return training


class _CurriculumProgress(BaseCallback):
    def _on_training_start(self) -> None:
        self._start_steps = self.model.num_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        total = max(int(self.model._total_timesteps) - self._start_steps, 1)
        progress = min(1.0, max(0.0, (self.num_timesteps - self._start_steps) / total))
        self.training_env.env_method("set_curriculum_progress", progress)
        self.logger.record("curriculum/progress", progress)


class _RecoveryEvaluation(BaseCallback):
    """Compare models under fixed seeds/strength, including the full standing tail."""

    def __init__(self, curriculum: str, best_path: Path, interval: int = 2_560_000, episodes: int = 8) -> None:
        super().__init__()
        self.curriculum = curriculum
        self.best_path = best_path
        self.interval = interval
        self.episodes = episodes
        self._best = (-1., -1., float("-inf"))

    def _on_training_start(self) -> None:
        self._last_eval = self.num_timesteps
        self._evaluate()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_eval >= self.interval:
            self._last_eval = self.num_timesteps
            self._evaluate()

    def _evaluate(self) -> None:
        from dataclasses import replace
        from robot_rl.strikes import StrikeSchedule
        from robot_rl.robot import SHOT_MASS, SHOT_SPEED

        env = HumanoidBalanceEnv(curriculum=self.curriculum)
        # Fixed impact strength; phase tracking has no regularization schedule.
        env.curriculum_progress = 1.0
        if self.curriculum == "impact":
            base = SHOT_MASS * SHOT_SPEED
            fixed = replace(env.curriculum.strikes, momentum=(base, base), momentum_end=None)
            env._schedule = StrikeSchedule(fixed)
        successes = 0
        styles, jitters = [], []
        try:
            for episode in range(self.episodes):
                options = None if self.curriculum == "impact" else {"pose": ("stand", "prone", "supine", "random")[episode % 4]}
                obs, _ = env.reset(seed=10_000 + episode, options=options)
                previous = None
                for _ in range(env.max_episode_steps):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    styles.append(info.get("motion_match", {}).get("imitation", 0.))
                    if previous is not None:
                        jitters.append(float(((action - previous) ** 2).mean()) ** .5)
                    previous = action.copy()
                    if terminated or truncated:
                        successes += bool(info.get("is_success", False))
                        break
            rate = successes / self.episodes
            style = sum(styles) / max(len(styles), 1)
            jitter = sum(jitters) / max(len(jitters), 1)
            self.logger.record("eval/recovery_success_rate", rate)
            self.logger.record("eval/motion_imitation", style)
            self.logger.record("eval/action_delta_rms", jitter)
            # First preserve recovery, then trade reference quality against jitter.
            quality = style - .25 * jitter
            self.logger.record("eval/motion_quality", quality)
            score = (rate, quality, -jitter)
            if score > self._best:
                self._best = score
                self.best_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_path))
                print(f"{self.curriculum} 固定评估：成功 {rate:.1%}，参考匹配 {style:.3f}，动作变化 {jitter:.3f}；保存 {self.best_path.name}。")
        finally:
            env.close()


class _StopOnSuccess(BaseCallback):
    def __init__(self, curriculum: str, best_path: Path) -> None:
        super().__init__()
        self.curriculum = curriculum
        self.best_path = best_path
        self.threshold = success_threshold(curriculum)
        self._flags: deque[bool] = deque(maxlen=SUCCESS_WINDOW)
        self._best = -1.0

    def _on_step(self) -> bool:
        if self.threshold is None:
            return True
        dones = self.locals.get("dones")
        infos = self.locals.get("infos") or []
        if dones is None:
            return True
        for done, info in zip(dones, infos):
            if done and "is_success" in info:
                self._flags.append(bool(info["is_success"]))
        if ready_to_stop(self._flags, threshold=self.threshold):
            rate = sum(self._flags) / len(self._flags)
            self._save_best(rate)
            print(f"课程 {self.curriculum} 最近 {SUCCESS_WINDOW} 局成功率 {rate:.2%}，提前停止。")
            return False
        return True

    def _on_rollout_end(self) -> None:
        if self.threshold is None or len(self._flags) < SUCCESS_WINDOW:
            return
        rate = sum(self._flags) / len(self._flags)
        self.logger.record("rollout/success_rate", rate)
        if rate > self._best and rate >= 0.5:
            self._save_best(rate)

    def _save_best(self, rate: float) -> None:
        if rate <= self._best:
            return
        improved = rate - self._best
        self._best = rate
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.best_path))
        if improved >= 0.01:
            print(f"课程 {self.curriculum} 最近 {SUCCESS_WINDOW} 局成功率 {rate:.2%}，写入 {self.best_path.name}。")


class _RewardCurves(BaseCallback):
    """奖励和动作曲线凑满两局再上报。短于一局会把局末的成败切成锯齿。"""

    def __init__(self, episode_steps: int) -> None:
        super().__init__()
        if episode_steps < 1:
            raise ValueError(f"上报窗口至少 1 步，收到 {episode_steps}")
        self._episode_steps = int(episode_steps)
        self._vec_steps = 0
        self._counts = 0
        self._sums: dict[str, float] = {}
        self._diagnostics: dict[str, float] = {}

    def _on_training_start(self) -> None:
        # SwanLab 会在自己的 _on_training_start 里换掉 logger。这里排在它后面，包的是新 logger。
        _use_chinese_metric_names(self.logger)

    def _on_step(self) -> bool:
        self._vec_steps += 1
        infos = self.locals.get("infos") or []
        for info in infos:
            terms = info.get("reward_terms")
            if not terms:
                continue
            self._counts += 1
            for key, value in terms.items():
                self._sums[key] = self._sums.get(key, 0.0) + float(value)
            metrics = {f"motion/{key}": value for key, value in info.get("motion_match", {}).items()}
            if "recovery_stage" in info:
                metrics["recovery/holding_fraction"] = float(info.get("recovery_stage") in ("hold", "complete"))
            for key, value in metrics.items():
                self._diagnostics[key] = self._diagnostics.get(key, 0.0) + float(value)
        return True

    def _on_rollout_end(self) -> None:
        if self._counts == 0 or self._vec_steps < self._episode_steps:
            return
        for key, total in self._sums.items():
            self.model.logger.record(f"reward/{key}", total / self._counts)
        for key, total in self._diagnostics.items():
            self.model.logger.record(key, total / self._counts)
        self._vec_steps = 0
        self._counts = 0
        self._sums.clear()
        self._diagnostics.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="训练两足人形抗击打和爬起")
    parser.add_argument("--curriculum", choices=tuple(CURRICULA), default="stand")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=N_ENVS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto 在有 CUDA 时用 GPU 做策略更新。")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--load", type=Path, default=None, help="从该检查点续训。--timesteps 仍是总步数，只补差额。")
    args = parser.parse_args()
    save_path = RUN_DIR / f"humanoid_ppo_{args.curriculum}" if args.save is None else args.save
    log_dir = save_path.parent / "swanlog"
    print(
        f"课程 {args.curriculum}。曲线在 {DASHBOARD_URL} 。"
        "曲线名是中文：训练/价值损失、采样/成功率、奖励/ 下是各项奖励。"
    )
    threading.Thread(target=_open_dashboard_when_ready, args=(log_dir,), daemon=True).start()
    train(
        timesteps=args.timesteps,
        n_envs=args.envs,
        seed=args.seed,
        device=args.device,
        save_path=save_path,
        curriculum=args.curriculum,
        load=args.load,
    )


def _drop_unsupported_gpu_power() -> None:
    """这张卡读不了功耗。失败的采集项丢掉，避免每个采样周期都打一条错误。"""
    from swanlab.sdk.internal.pkg import safe
    from swanlab.sdk.internal.probe_python.protocol import CollectorProtocol

    try:
        from swanlab.vendor import pynvml
    except Exception:
        return
    if getattr(CollectorProtocol, "_robot_rl_collect", False):
        return

    def collect(self):
        results = []
        kept = []
        for key, handler in self._handlers:
            try:
                results.append((key, handler()))
                kept.append((key, handler))
            except pynvml.NVMLError as exc:
                if "Not Supported" in str(exc):
                    continue
                kept.append((key, handler))
                with safe.block(message=f"Failed to collect while calling {key}"):
                    raise exc
            except Exception as exc:
                kept.append((key, handler))
                with safe.block(message=f"Failed to collect while calling {key}"):
                    raise exc
        self._handlers = kept
        return results

    CollectorProtocol.collect = collect
    CollectorProtocol._robot_rl_collect = True


def _open_dashboard_when_ready(log_dir: Path) -> None:
    database = log_dir / "runs.swanlab"
    deadline = time.time() + 180
    while not database.exists() and time.time() < deadline:
        time.sleep(0.25)
    if not database.exists():
        return
    _open_dashboard(log_dir)
    webbrowser.open(DASHBOARD_URL)


def _board_is_ready() -> bool:
    try:
        with urllib.request.urlopen(DASHBOARD_URL + "/api/v1/project", timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return payload.get("code") == 0


def _open_dashboard(log_dir: Path) -> None:
    if _board_is_ready():
        return
    probe = socket.socket()
    probe.settimeout(0.3)
    try:
        probe.connect(("127.0.0.1", 5092))
        return
    except OSError:
        pass
    finally:
        probe.close()

    log_dir.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    log_file = open(log_dir.parent / "swanlab-dashboard.log", "ab")
    subprocess.Popen(
        [sys.executable, "-m", "swanlab", "watch", str(log_dir), "--host", "127.0.0.1", "--port", "5092"],
        cwd=os.getcwd(),
        stdout=log_file,
        stderr=log_file,
        env=env,
        creationflags=flags,
        close_fds=False,
    )
    log_file.close()


if __name__ == "__main__":
    main()
