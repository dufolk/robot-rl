"""一个进程里串行推多个人形，再开若干个这样的进程。

和 Unity 里单线程跑几十个环境是同一种排法：人与人之间没有管道，一个循环挨个 `step`。
进程数按核数来，默认总人数是 256。查看器不走这里，仍然只显示一个。
"""

from __future__ import annotations

import os
import traceback
from multiprocessing import get_context
from multiprocessing.connection import Connection

# 子进程重新 import 时也只留一条计算线程，避免和仿真抢核。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

from robot_rl.env import HumanoidBalanceEnv

N_ENVS = 256


def worker_count(n_envs: int) -> int:
    cores = os.cpu_count() or 8
    return max(1, min(int(n_envs), max(1, cores - 4)))


class ShardedVecEnv(VecEnv):
    def __init__(self, n_envs: int, curriculum: str, seed: int = 0, n_workers: int | None = None) -> None:
        if n_envs < 1:
            raise ValueError(f"环境数至少为 1，收到 {n_envs}")
        self._n_workers = worker_count(n_envs) if n_workers is None else int(n_workers)
        if self._n_workers < 1 or self._n_workers > n_envs:
            raise ValueError(f"进程数要在 1 和 {n_envs} 之间，收到 {self._n_workers}")
        counts = [n_envs // self._n_workers] * self._n_workers
        for index in range(n_envs % self._n_workers):
            counts[index] += 1
        self._counts = counts
        self._offsets = [0]
        for count in counts[:-1]:
            self._offsets.append(self._offsets[-1] + count)

        ctx = get_context("spawn")
        self._ctx = ctx
        self._remotes: list[Connection] = []
        self._processes = []
        rank = 0
        for count in counts:
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=_serve,
                args=(child, count, curriculum, seed + rank),
                daemon=True,
            )
            process.start()
            child.close()
            self._remotes.append(parent)
            self._processes.append(process)
            rank += count

        observation_space, action_space = self._call(0, ("spec", None))
        super().__init__(n_envs, observation_space, action_space)
        self._waiting = False

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions)
        for remote, offset, count in zip(self._remotes, self._offsets, self._counts):
            remote.send(("step", actions[offset : offset + count]))
        self._waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        if not self._waiting:
            raise RuntimeError("要先 step_async")
        batches = [self._recv(remote) for remote in self._remotes]
        self._waiting = False
        observations = np.concatenate([batch[0] for batch in batches])
        rewards = np.concatenate([batch[1] for batch in batches])
        dones = np.concatenate([batch[2] for batch in batches])
        infos: list[dict] = []
        for batch in batches:
            infos.extend(batch[3])
        return observations, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        if self._waiting:
            for remote in self._remotes:
                self._recv(remote)
            self._waiting = False
        payloads = []
        for offset, count in zip(self._offsets, self._counts):
            seeds = self._seeds[offset : offset + count]
            options = self._options[offset : offset + count]
            payloads.append((seeds, options))
        for remote, payload in zip(self._remotes, payloads):
            remote.send(("reset", payload))
        batches = [self._recv(remote) for remote in self._remotes]
        self._reset_seeds()
        self._reset_options()
        observations = np.concatenate([batch[0] for batch in batches])
        self.reset_infos = []
        for batch in batches:
            self.reset_infos.extend(batch[1])
        return observations

    def close(self) -> None:
        for remote, process in zip(self._remotes, self._processes):
            if process.is_alive():
                try:
                    remote.send(("close", None))
                    self._recv(remote)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            remote.close()
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list:
        return self._query("get_attr", attr_name, indices)

    def set_attr(self, attr_name: str, value, indices: VecEnvIndices = None) -> None:
        grouped, _places = self._group(indices)
        for remote, local in zip(self._remotes, grouped):
            remote.send(("set_attr", (attr_name, value, local)))
        for remote in self._remotes:
            self._recv(remote)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list:
        payload = (method_name, method_args, method_kwargs)
        return self._query("env_method", payload, indices)

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None) -> list[bool]:
        return self._query("is_wrapped", wrapper_class, indices)

    def get_images(self) -> list[None]:
        return [None] * self.num_envs

    def _query(self, command: str, extra: object, indices: VecEnvIndices) -> list:
        grouped, places = self._group(indices)
        for remote, local in zip(self._remotes, grouped):
            remote.send((command, (extra, local)))
        answers = [self._recv(remote) for remote in self._remotes]
        result: list = [None] * sum(len(place) for place in places)
        for place, answer in zip(places, answers):
            for slot, value in zip(place, answer):
                result[slot] = value
        return result

    def _group(self, indices: VecEnvIndices) -> tuple[list[list[int]], list[list[int]]]:
        chosen = list(self._get_indices(indices))
        grouped = [[] for _ in self._counts]
        places = [[] for _ in self._counts]
        for place, index in enumerate(chosen):
            worker, local = self._locate(index)
            grouped[worker].append(local)
            places[worker].append(place)
        return grouped, places

    def _locate(self, index: int) -> tuple[int, int]:
        for worker, offset in enumerate(self._offsets):
            if index < offset + self._counts[worker]:
                return worker, index - offset
        raise IndexError(index)

    def _call(self, worker: int, message: tuple) -> object:
        self._remotes[worker].send(message)
        return self._recv(self._remotes[worker])

    @staticmethod
    def _recv(remote: Connection) -> object:
        message = remote.recv()
        if (
            isinstance(message, tuple)
            and len(message) == 2
            and isinstance(message[0], str)
            and message[0] == "__error__"
        ):
            raise RuntimeError(message[1])
        return message


def _serve(conn: Connection, n_local: int, curriculum: str, seed: int) -> None:
    try:
        envs = []
        for rank in range(n_local):
            env = HumanoidBalanceEnv(curriculum=curriculum)
            env.action_space.seed(seed + rank)
            env = Monitor(env, info_keywords=("is_success",))
            envs.append(env)
        while True:
            command, payload = conn.recv()
            if command == "spec":
                conn.send((envs[0].observation_space, envs[0].action_space))
            elif command == "step":
                conn.send(_step_all(envs, payload))
            elif command == "reset":
                conn.send(_reset_all(envs, payload, seed))
            elif command == "get_attr":
                name, local = payload
                conn.send([envs[index].get_wrapper_attr(name) for index in local])
            elif command == "set_attr":
                name, value, local = payload
                for index in local:
                    setattr(envs[index].unwrapped, name, value)
                conn.send(True)
            elif command == "env_method":
                (method_name, method_args, method_kwargs), local = payload
                conn.send(
                    [
                        envs[index].get_wrapper_attr(method_name)(*method_args, **method_kwargs)
                        for index in local
                    ]
                )
            elif command == "is_wrapped":
                wrapper_class, local = payload
                conn.send([_is_wrapped(envs[index], wrapper_class) for index in local])
            elif command == "close":
                for env in envs:
                    env.close()
                conn.send(True)
                break
            else:
                raise RuntimeError(f"未知命令 {command}")
    except BaseException:
        try:
            conn.send(("__error__", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        conn.close()


def _step_all(envs: list, actions: np.ndarray) -> tuple:
    observations = []
    rewards = np.zeros(len(envs), dtype=np.float32)
    dones = np.zeros(len(envs), dtype=bool)
    infos = []
    for index, env in enumerate(envs):
        obs, reward, terminated, truncated, info = env.step(actions[index])
        done = bool(terminated or truncated)
        info = dict(info)
        info["TimeLimit.truncated"] = bool(truncated and not terminated)
        if done:
            info["terminal_observation"] = obs
            obs, _reset_info = env.reset()
        observations.append(np.asarray(obs, dtype=np.float32))
        rewards[index] = float(reward)
        dones[index] = done
        infos.append(info)
    return np.stack(observations), rewards, dones, infos


def _reset_all(envs: list, payload: tuple, seed: int) -> tuple:
    seeds, options = payload
    observations = []
    infos = []
    for index, env in enumerate(envs):
        env_seed = seeds[index]
        if env_seed is None:
            env_seed = seed + index
        kwargs = {"options": options[index]} if options[index] else {}
        obs, info = env.reset(seed=env_seed, **kwargs)
        observations.append(np.asarray(obs, dtype=np.float32))
        infos.append(info)
    return np.stack(observations), infos


def _is_wrapped(env, wrapper_class) -> bool:
    current = env
    while current is not None:
        if isinstance(current, wrapper_class):
            return True
        current = getattr(current, "env", None)
    return False
