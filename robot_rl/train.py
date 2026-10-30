"""PPO 训练主循环。

用 `--curriculum` 选课程。getup 默认自主发现起身，可选第二阶段精修，impact 继续抗击打。
getup 和 impact 跑满预算，按固定协议评估并单独保存最佳模型。
HumanUP 保留身体输入并追加三步历史，共 384 维，策略为 256/256 ELU MLP。
有 GPU 时更新放在 GPU 上。观测里没有炮弹，策略只能在被打中之后恢复，不能提前躲开。
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import deque
from pathlib import Path
from time import perf_counter

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
from robot_rl.checkpoint import read_metadata
from robot_rl.env import HumanoidBalanceEnv
from robot_rl.humanup import MODEL_CONTRACT, initial_action_std, exploration_scale
from robot_rl.humanup_ppo import HumanUPPPO, DESIRED_KL
from robot_rl.humanup_policy import RecoveryPolicy
from robot_rl.vec import N_ENVS, ShardedVecEnv, worker_count

RUN_DIR = Path("runs")
# 检查点大约每十几万步存一次。只留最近几个，避免 runs/checkpoints 堆到几百个。
KEEP_CHECKPOINTS = 5
# Stand retains its existing PPO configuration. HumanUP uses the official
# adaptive 2e-4 learning rate and desired KL=.008 below.
LEARNING_RATE = 1e-4
N_EPOCHS = 4
TARGET_KL = 3.5
BATCH_SIZE = 2048
LOG_STD_INIT = -1.5
# 代码里的键保持英文。看板只在上报时换成中文。
METRIC_NAMES = {
    "curriculum/progress": "课程/进度",
    "curriculum/prone_fraction": "课程/趴地比例",
    "reward/orientation": "奖励/躯干倾斜",
    "rollout/ep_rew_mean": "采样/平均回报",
    "rollout/ep_len_mean": "采样/平均局长",
    "rollout/success_rate": "采样/成功率",
    "time/fps": "时间/每秒步数",
    "time/iterations": "时间/轮数",
    "time/time_elapsed": "时间/已用秒",
    "time/total_timesteps": "时间/总步数",
    "time/sampling_seconds": "时间/本轮采样秒数",
    "time/update_seconds": "时间/本轮更新秒数",
    "time/sampling_fps": "时间/采样每秒步数",
    "time/sampling_fraction": "时间/采样阶段占比",
    "train/approx_kl": "训练/KL",
    "train/analytic_kl": "训练/解析KL",
    "train/proposed_kl": "训练/回滚前KL",
    "train/learning_rate_ceiling": "训练/学习率上限",
    "train/update_rejected": "训练/本轮回滚",
    "train/rejected_updates": "训练/累计回滚",
    "train/target_angle_std": "训练/目标角标准差",
    "train/latent_target_std": "训练/压缩前目标角标准差",
    "train/action_edge_fraction": "训练/动作近限位比例",
    "train/entropy_coefficient": "训练/熵系数",
    "train/weighted_entropy_loss": "训练/加权熵项",
    "train/policy_gradient_norm": "训练/策略梯度范数",
    "train/entropy_gradient_norm": "训练/熵梯度范数",
    "train/entropy_gradient_ratio": "训练/熵策略梯度比",
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
    "eval/stand_success_rate": "评估/站立成功率",
    "eval/supine_success_rate": "评估/仰躺起身成功率",
    "eval/prone_success_rate": "评估/趴地起身成功率",
    "eval/random_success_rate": "评估/随机姿势成功率",
    "eval/impact_success_rate": "评估/受击恢复成功率",
    "eval/motion_tracking": "评估/轨迹跟踪",
    "eval/action_delta_rms": "评估/动作变化",
    "eval/motion_quality": "评估/动作质量",
    "recovery/holding_fraction": "恢复/保持比例",
    "motion/imitation": "动作/模仿",
    "motion/reference_pose": "动作/参考姿态",
    "motion/reference_velocity": "动作/参考速度",
    "motion/reference_ee": "动作/参考手脚",
    "motion/reference_com": "动作/参考质心",
    "reward/total": "奖励/合计",
    "reward/height_deficit": "奖励/站高差距",
    "reward/upright_deficit": "奖励/直立差距",
    "reward/support_deficit": "奖励/支撑差距",
    "reward/progress": "奖励/恢复进展",
    "reward/stable": "奖励/静止站稳",
    "curriculum/assist_fraction": "课程/当前辅助体重比",
    "curriculum/assistance_level": "课程/撤除辅助级别",
    "reward/assist": "诊断/拉力",
    "reward/imitation": "奖励/模仿",
    "reward/head_height": "奖励/头部高度",
    "reward/tracking": "奖励/关节轨迹",
    "reward/head_tracking": "奖励/头部轨迹",
    "reward/symmetry": "奖励/身体对称",
    "reward/waist_symmetry": "奖励/腰部对称",
    "reward/standing_pose": "奖励/站姿关节",
    "reward/feet_distance": "奖励/脚间距",
    "reward/feet_orientation": "奖励/脚倾斜",
    "reward/feet_height": "奖励/脚高度",
    "recovery/regularization_strength": "恢复/正则强度",
    "motion/tracking": "动作/轨迹跟踪",
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
    "reward/standing": "奖励/严格站稳",
    "reward/time_cost": "奖励/未站稳时间成本",
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
    if curriculum in ("getup", "impact"):
        return None
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
    humanup_stage: str = "discovery",
    reference: Path | None = None,
) -> PPO:
    import torch

    torch.set_num_threads(1)
    _drop_unsupported_gpu_power()
    course = get_curriculum(curriculum)
    humanup = course.name in ("getup", "impact")
    env_kwargs = {"humanup_stage": humanup_stage, "reference_path": None if reference is None else str(reference.resolve())}
    # Stage II prerequisites must be validated before starting worker processes.
    probe = HumanoidBalanceEnv(curriculum=course.name, **env_kwargs)
    horizon = probe.max_episode_steps
    action_std = initial_action_std(probe.sim) if humanup else None
    action_scale = (exploration_scale(probe.sim) if humanup else probe.sim.action_scale).copy()
    action_scale_low, action_scale_high = probe.sim.action_scale_low.copy(), probe.sim.action_scale_high.copy()
    probe.close()
    prefix = f"humanoid_ppo_{course.name}"
    save_path = RUN_DIR / prefix if save_path is None else Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps = (24 if humanup else 64) if n_steps is None else int(n_steps)
    batch_size = max(2, steps * n_envs // 4) if humanup else min(BATCH_SIZE, steps * n_envs)
    ent_coef = .001 if humanup else 0.
    learning_rate = 2e-4 if humanup else LEARNING_RATE
    target_kl = DESIRED_KL if humanup else TARGET_KL
    epochs = 5 if humanup else N_EPOCHS
    algorithm = HumanUPPPO if humanup else PPO
    # Validate before spawning hundreds of environments. Different observations
    # define a new policy; silently padding old input weights is not a resume.
    if load is not None:
        probe = HumanoidBalanceEnv(curriculum=course.name, **env_kwargs)
        try:
            metadata = read_metadata(load, ("observation_space", "action_space", "humanup_contract"))
            if metadata["observation_space"] != probe.observation_space or metadata["action_space"] != probe.action_space:
                raise ValueError(
                    f"检查点空间 {metadata['observation_space'].shape}/{metadata['action_space'].shape} "
                    f"与本课程 {probe.observation_space.shape}/{probe.action_space.shape} 不兼容。"
                    "HumanUP 保留 189 维并追加三帧历史，共 384 维。旧布局不能直接续训；请从头训练或加载新版 HumanUP。"
                )
            if humanup and metadata["humanup_contract"] != MODEL_CONTRACT:
                raise ValueError("检查点的动作分布/奖励契约不兼容，不能直接续训。请从头训练；旧日志和权重保留。")
        finally:
            probe.close()
    workers = 1 if n_envs == 1 else worker_count(n_envs)
    print(
        f"并行 {n_envs} 个人形，{workers} 个进程各自串行推动，batch {batch_size}，"
        f"单轮 KL 超过 {1.5 * target_kl:g} 就停下，device {device}。"
    )
    if n_envs == 1:
        vec = make_vec_env(
            HumanoidBalanceEnv,
            n_envs=1,
            seed=seed,
            env_kwargs={"curriculum": course.name, **env_kwargs},
            vec_env_cls=DummyVecEnv,
            monitor_kwargs={"info_keywords": ("is_success",)},
        )
    else:
        vec = ShardedVecEnv(n_envs, curriculum=course.name, seed=seed, n_workers=workers, env_kwargs=env_kwargs, compact_infos=True)
    if load is None:
        model = algorithm(
            RecoveryPolicy if humanup else "MlpPolicy",
            vec,
            learning_rate=learning_rate,
            n_steps=steps,
            batch_size=batch_size,
            n_epochs=epochs,
            target_kl=target_kl,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=1. if humanup else .5,
            clip_range_vf=.2 if humanup else None,
            max_grad_norm=1. if humanup else .5,
            seed=seed,
            device=device,
            verbose=1,
            policy_kwargs={
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                "activation_fn": torch.nn.ELU if humanup else torch.nn.Tanh,
                "log_std_init": LOG_STD_INIT,
            },
        )
        if humanup:
            with torch.no_grad():
                model.policy.log_std.copy_(torch.as_tensor(action_std, device=model.device).log())
        already = 0
    else:
        model = algorithm.load(
            str(load),
            env=vec,
            device=device,
            learning_rate=learning_rate,
            n_steps=steps,
            batch_size=batch_size,
            n_epochs=epochs,
            target_kl=target_kl,
        )
        already = int(model.num_timesteps)
        print(f"从 {load} 续训，已有 {already} 步。熵系数按本课程设为 {ent_coef}。")
    model.ent_coef = ent_coef
    if humanup:
        model.policy.log_std.requires_grad_(True)
        model.humanup_contract = MODEL_CONTRACT
        model.humanup_action_scale = action_scale.tolist()
        model.humanup_action_scale_low = action_scale_low.tolist()
        model.humanup_action_scale_high = action_scale_high.tolist()
        model.getup_reward = "recovery-deficit-assistance" if humanup_stage == "discovery" else "humanup-refinement"
        model.humanup_stage = humanup_stage
        model.humanup_reference = env_kwargs["reference_path"]
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
            experiment_name=course.name,
            description=course.description,
            log_dir=str(log_dir),
            mode="local",
        ),
        _RewardCurves(2 * horizon),
        _SamplingTiming(),
        _PolicyDriftFromInit(),
        _CurriculumProgress(),
        _StopOnSuccess(course.name, save_path.parent / "checkpoints" / f"{prefix}_best.zip"),
    ]
    if course.name in ("getup", "impact"):
        callbacks.append(_RecoveryEvaluation(course.name, save_path.parent / "checkpoints" / f"{prefix}_best.zip", env_kwargs=env_kwargs))
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


class _SamplingTiming(BaseCallback):
    def _on_rollout_start(self):
        self._started = perf_counter()
        self._start_steps = self.num_timesteps

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        elapsed = max(perf_counter() - self._started, 1e-9)
        self.model._sampling_seconds = elapsed
        self.logger.record("time/sampling_seconds", elapsed)
        self.logger.record("time/sampling_fps", (self.num_timesteps - self._start_steps) / elapsed)


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

    def __init__(self, curriculum: str, best_path: Path, interval: int = 2_560_000, episodes: int = 8, env_kwargs: dict | None = None) -> None:
        super().__init__()
        self.curriculum = curriculum
        self.best_path = best_path
        self.interval = interval
        self.episodes = episodes
        self.env_kwargs = env_kwargs or {}
        self._best = (-1., -1., -1., float("-inf"))

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

        env = HumanoidBalanceEnv(curriculum=self.curriculum, **self.env_kwargs)
        # Evaluate full regularization and fixed impact strength.
        env.curriculum_progress = 1.0
        env._reward.adapt_regularization = False
        if self.curriculum == "impact":
            base = SHOT_MASS * SHOT_SPEED
            fixed = replace(env.curriculum.strikes, momentum=(base, base), momentum_end=None)
            env._schedule = StrikeSchedule(fixed)
        successes = 0
        pose_outcomes = {}
        styles, jitters, postures = [], [], []
        try:
            for episode in range(self.episodes):
                options = None if self.curriculum == "impact" else {"pose": ("stand", "prone", "supine", "random")[episode % 4]}
                pose = "impact" if options is None else options["pose"]
                obs, _ = env.reset(seed=10_000 + episode, options=options)
                previous = None
                for _ in range(env.max_episode_steps):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    styles.append(info.get("motion_match", {}).get("tracking", 0.))
                    postures.append(min(info["head_height"] / info["stand_head_height"], 1.) * max(info["upright"], 0.))
                    if previous is not None:
                        jitters.append(float(((action - previous) ** 2).mean()) ** .5)
                    previous = action.copy()
                    if terminated or truncated:
                        success = bool(info.get("is_success", False))
                        successes += success
                        pose_outcomes.setdefault(pose, []).append(success)
                        break
            rate = successes / self.episodes
            for pose, outcomes in pose_outcomes.items():
                self.logger.record(f"eval/{pose}_success_rate", sum(outcomes) / len(outcomes))
            fallen = [result for pose, outcomes in pose_outcomes.items() if pose != "stand" for result in outcomes]
            recovery_rate = sum(fallen) / max(len(fallen), 1)
            style = sum(styles) / max(len(styles), 1)
            jitter = sum(jitters) / max(len(jitters), 1)
            self.logger.record("eval/recovery_success_rate", rate)
            self.logger.record("eval/motion_tracking", style)
            self.logger.record("eval/action_delta_rms", jitter)
            # First preserve recovery, then trade reference quality against jitter.
            quality = sum(postures) / max(len(postures), 1) - .25 * jitter
            self.logger.record("eval/motion_quality", quality)
            # Standing starts alone must not outrank a model that learns to
            # recover. Keep the combined rate visible as a separate metric.
            score = (recovery_rate, rate, quality, -jitter)
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
            for name in ("assist_fraction", "assistance_level"):
                if name in info:
                    metrics[f"curriculum/{name}"] = info[name]
            if "prone_probability" in info:
                metrics["curriculum/prone_fraction"] = info["prone_probability"]
            if "regularization_strength" in info:
                metrics["recovery/regularization_strength"] = info["regularization_strength"]
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
    # Redirected Windows terminals otherwise use GBK, which cannot encode
    # SwanLab's status symbols and aborts training before the first rollout.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="训练两足人形抗击打和爬起")
    parser.add_argument("--curriculum", choices=tuple(CURRICULA), default="stand")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=N_ENVS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto 在有 CUDA 时用 GPU 做策略更新。")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--stage", choices=("discovery", "refinement"), default="discovery")
    parser.add_argument("--reference", type=Path, default=None, help="第二阶段使用本机器人成功轨迹导出的 npz")
    parser.add_argument("--load", type=Path, default=None, help="从该检查点续训。--timesteps 仍是总步数，只补差额。")
    args = parser.parse_args()
    save_path = args.save
    print(
        f"课程 {args.curriculum}。SwanLab 日志保存在 swanlog，按需手动打开看板。"
        "曲线名是中文：训练/价值损失、采样/成功率、奖励/ 下是各项奖励。"
    )
    train(
        timesteps=args.timesteps,
        n_envs=args.envs,
        seed=args.seed,
        device=args.device,
        save_path=save_path,
        curriculum=args.curriculum,
        load=args.load,
        humanup_stage=args.stage,
        reference=args.reference,
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


if __name__ == "__main__":
    main()
