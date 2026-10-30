"""HumanUP adaptations: physical history and robot-generated Stage I trajectories.

No phase/skill/reference target is part of an actor observation. Stage II keeps
the upstream time-indexed reward explicitly; history does not guarantee perfect
inference of that hidden time. ROA/privileged latents are not fabricated here.
"""
from pathlib import Path
from collections import deque

import numpy as np

from robot_rl.robot import JOINTS
from robot_rl.motion import physical_observation, physical_observation_size, physical_observation_slices

HISTORY_STEPS = 3
FORMAT = "robot_rl_humanup_discovery_v1"
MODEL_CONTRACT = "humanup-v4-history3"
JOINT_LIMIT_CONTRACTS = {"humanup-v3-bounded-progress", MODEL_CONTRACT}
SCALED_CONTRACTS = {"humanup-v2-scaled-dt-1khz", *JOINT_LIMIT_CONTRACTS}
PHYSICS_DT = .001
ACTION_MAPPING = "joint_limits"
ANGULAR_SCALE = .25
JOINT_VELOCITY_SCALE = .05


def exploration_scale(sim):
    if sim.action_mapping == ACTION_MAPPING:
        lower, upper = sim.action_scale_low, sim.action_scale_high
        # Variance of a zero-mean Gaussian mapped with different slopes on
        # each side, before the small tanh contraction at initialization.
        return np.sqrt(.5 * (lower ** 2 + upper ** 2) - (upper - lower) ** 2 / (2. * np.pi))
    return sim.action_scale


def initial_action_std(sim):
    """Transfer upstream exploration in target-angle units to our action scale.

    G1 uses actor sigma * action_scale(.5); our normalized [-1,1] convention
    instead uses a different angle span at each joint. Do not copy raw sigmas.
    """
    angle_std = np.array([.05 if j.name.startswith("waist") else
                          .2 if "knee" in j.name else
                          .15 if "hip" in j.name else .1 for j in JOINTS])
    return angle_std / np.maximum(exploration_scale(sim), 1e-6)


def proprioception(sim, normalized=True):
    """Upstream proprioception: angular velocity, roll/pitch, q, qd, last action.

    65 rather than G1's 74 dimensions because this robot has 20 actuators.
    """
    rotation = sim.data.xmat[sim.pelvis_id].reshape(3, 3)
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2]))
    _, angular = sim._base_velocity()
    qd = sim.data.qvel[sim.dof_adr]
    if normalized:
        angular = angular * ANGULAR_SCALE
        qd = qd * JOINT_VELOCITY_SCALE
    values = np.r_[angular, roll, pitch, sim.data.qpos[sim.qpos_adr] - sim.default_q,
                   qd, sim.last_action].astype(np.float32)
    return np.clip(values, -100., 100.) if normalized else values


def normalized_state_observation(sim):
    """Keep all 189 features; apply fixed scales consistently at train/deploy."""
    values = physical_observation(sim).copy()
    layout = physical_observation_slices(sim)
    values[layout["base_ang_vel"]] *= ANGULAR_SCALE
    bodies = values[layout["body_states"]].reshape(-1, 13)
    bodies[:, 10:13] *= ANGULAR_SCALE
    return np.clip(values, -100., 100.)


def observation_size(sim):
    return physical_observation_size(sim) + HISTORY_STEPS * (5 + 3 * sim.n_act)


class ObservationHistory:
    def __init__(self, normalized=True, steps=HISTORY_STEPS):
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError("History must contain at least one frame")
        self.past = deque(maxlen=self.steps)
        self.current = None
        self.normalized = normalized

    def reset(self, sim):
        self.current = proprioception(sim, self.normalized)
        self.past.clear()
        self.past.extend(self.current.copy() for _ in range(self.steps))

    def advance(self, sim):
        if self.current is None:
            self.reset(sim)
        else:
            self.past.append(self.current.copy())
            self.current = proprioception(sim, self.normalized)

    def observation(self, sim):
        if self.current is None:
            raise RuntimeError("Observation history must be reset before inference")
        state = normalized_state_observation(sim) if self.normalized else physical_observation(sim)
        return np.concatenate((state, np.asarray(self.past).ravel()))


class DiscoveryPoseCurriculum:
    """Learn canonical supine recovery before adding prone cases to one policy.

    Upstream uses separate stand-up / roll-over policies. This explicitly keeps
    our single-policy adaptation, without treating mixed hard starts as a replica.
    """
    def __init__(self):
        self.supine_outcomes = deque(maxlen=64)
        self.prone_probability = 0.

    def update(self, pose, success):
        if pose == "supine":
            self.supine_outcomes.append(bool(success))
            if len(self.supine_outcomes) >= 16 and np.mean(self.supine_outcomes) >= .8:
                self.prone_probability = min(.45, self.prone_probability + .025)

    def sample(self, rng):
        return str(rng.choice(("stand", "supine", "prone"),
                              p=(.1, .9 - self.prone_probability, self.prone_probability)))


class GetupCurriculum:
    """Success-driven assistance withdrawal; no elapsed-step difficulty ramp."""
    def __init__(self):
        self.level = 0
        self.outcomes = deque(maxlen=16)
        self.prone_probability = 0.

    @property
    def assistance(self):
        return (.6, .4, .2, 0.)[self.level]

    def update(self, pose, success):
        if pose not in ("supine", "prone"):
            return
        self.outcomes.append(bool(success))
        if len(self.outcomes) >= 8 and np.mean(self.outcomes) >= .75:
            if self.level < 3:
                self.level += 1
            else:
                self.prone_probability = min(.4, self.prone_probability + .1)
            self.outcomes.clear()

    def sample(self, rng):
        return str(rng.choice(("stand", "supine", "prone"),
                             p=(.3, .7 - self.prone_probability, self.prone_probability)))


class DiscoveryTrajectory:
    """Only accept trajectories exported from this robot, never mocap retargets."""
    def __init__(self, path):
        if path is None:
            raise ValueError("HumanUP refinement requires --reference from a successful discovery rollout")
        self.path = Path(path).resolve()
        with np.load(self.path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        if str(data.get("format", "")) != FORMAT:
            raise ValueError("Expected a HumanUP discovery trajectory, not a DeepMimic asset")
        if list(data.get("joint_names", [])) != [j.name for j in JOINTS]:
            raise ValueError("HumanUP trajectory joint order does not match this robot")
        for pose in ("prone", "supine"):
            times, q, head = (data[f"{pose}_{key}"] for key in ("time", "q", "head_height"))
            if (times.ndim != 1 or len(times) < 2 or times[0] != 0.
                    or not np.all(np.diff(times) > 0.) or q.shape != (len(times), len(JOINTS))
                    or head.shape != times.shape or not all(np.isfinite(v).all() for v in (times, q, head))):
                raise ValueError(f"Invalid HumanUP trajectory: {pose}")
        self.data = data

    def duration(self, pose):
        return float(self.data[f"{pose}_time"][-1])

    def sample(self, pose, time):
        times = self.data[f"{pose}_time"]
        q = self.data[f"{pose}_q"]
        target = np.array([np.interp(time, times, q[:, j]) for j in range(q.shape[1])])
        head = float(np.interp(time, times, self.data[f"{pose}_head_height"]))
        return target, head
