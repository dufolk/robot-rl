"""Body-only policy observations; DeepMimic-style references are reward-only.

This deliberately omits the original paper's policy phase / skill conditioning.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import mujoco

REFERENCE_PATH = Path(__file__).parent / "assets/motions/getup_deepmimic.npz"
CLIPS = ("prone", "supine")


@lru_cache(maxsize=1)
def reference_library() -> dict[str, np.ndarray]:
    with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
        clips = {key: archive[key].copy() for key in archive.files}
    for values in clips.values():
        values.flags.writeable = False
    return clips


def heading_basis(sim):
    r = sim.data.xmat[sim.pelvis_id].reshape(3, 3)
    yaw = np.arctan2(r[1, 0], r[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def body_observation(sim) -> np.ndarray:
    """Physical body states only: no reference, clock, task or skill inputs."""
    basis = heading_basis(sim)
    root = sim.data.xpos[sim.pelvis_id]
    chunks = [np.array([root[2]])]
    for body in sim.robot_body_ids:
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, (basis.T @ sim.data.xmat[body].reshape(3, 3)).ravel())
        if quat[0] < 0.:
            quat *= -1.
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_BODY, int(body), velocity, 0)
        chunks.extend(((sim.data.xpos[body] - root) @ basis, quat,
                       velocity[3:] @ basis, velocity[:3] @ basis))
    return np.concatenate(chunks).astype(np.float32)


def fused_observation_size(sim) -> int:
    return sim.obs_size + 13 * len(sim.robot_body_ids)


def fused_observation(sim) -> np.ndarray:
    """Legacy proprioception/contact/history plus heading-local body states.

    The original prefix already contains root height, so omit that duplicate.
    Reference poses, skill IDs, phase, and curriculum progress are never read.
    """
    return np.concatenate((sim.observation(), body_observation(sim)[1:]))


def physical_observation_slices(sim) -> dict[str, slice]:
    """Single canonical layout; old 89/157/245 layouts are viewer-only here.

    Link orientations/velocities represent joint configuration/motion, so do
    not append joint q/qd or duplicate base velocity/gravity features. Retain
    contact and previous command, which cannot be recovered from a pose alone.
    This removes duplicate representations, not all kinematic constraints:
    body positions/orientations still obey the skeleton's forward kinematics.
    """
    fields = (("base_height", 1), ("base_orientation", 4),
              ("base_lin_vel", 3), ("base_ang_vel", 3),
              ("body_states", 13 * (len(sim.robot_body_ids) - 1)),
              ("last_action", sim.n_act), ("pressure", len(sim.pressure)))
    result, cursor = {}, 0
    for name, count in fields:
        result[name] = slice(cursor, cursor + count)
        cursor += count
    return result


def physical_observation_size(sim) -> int:
    return physical_observation_slices(sim)["pressure"].stop


def physical_observation(sim) -> np.ndarray:
    """Current body/contact state and previous action, without task guidance."""
    bodies = body_observation(sim)
    states = bodies[1:].reshape(-1, 13)
    root = np.flatnonzero(sim.robot_body_ids == sim.pelvis_id)[0]
    other = sim.robot_body_ids != sim.pelvis_id
    # Root-relative root position is identically zero: omit all three entries.
    return np.concatenate((bodies[:1], states[root, 3:], states[other].ravel(),
                           sim.last_action, sim.pressure)).astype(np.float32)


def tracking_features(sim) -> np.ndarray:
    """Heading-local EE / COM; remove root XY translation, retain world height."""
    basis = heading_basis(sim)
    origin = sim.data.xpos[sim.pelvis_id].copy()
    origin[2] = 0.
    ids = [mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, n)
           for n in ("right_foot_geom", "left_foot_geom", "right_hand", "left_hand")]
    ee = (sim.data.geom_xpos[ids] - origin) @ basis
    com = (sim.data.subtree_com[sim.pelvis_id] - origin) @ basis
    root_quat = np.empty(4)
    mujoco.mju_mat2Quat(root_quat, (basis.T @ sim.data.xmat[sim.pelvis_id].reshape(3, 3)).ravel())
    return np.r_[ee.ravel(), com, root_quat]


def sample_reference(clip: str, time: float) -> dict[str, np.ndarray]:
    bank = reference_library()
    times = bank[f"{clip}_time"]
    time = float(np.clip(time, 0., times[-1]))
    right = min(int(np.searchsorted(times, time, side="right")), len(times) - 1)
    left = max(0, right - 1)
    alpha = 0. if right == left else (time - times[left]) / (times[right] - times[left])
    result = {}
    for key in ("q", "velocity", "features", "qpos", "qvel"):
        values = bank[f"{clip}_{key}"]
        a, b = values[left].astype(float), values[right].astype(float).copy()
        if key == "qpos" and np.dot(a[3:7], b[3:7]) < 0.:
            b[3:7] *= -1.
        result[key] = (1. - alpha) * a + alpha * b
    result["qpos"][3:7] /= np.linalg.norm(result["qpos"][3:7])
    # Quaternion sign is arbitrary, including across adjacent source frames.
    qa = bank[f"{clip}_features"][left, 15:19]
    qb = bank[f"{clip}_features"][right, 15:19]
    if np.dot(qa, qb) < 0.:
        qb = -qb
    quat = (1. - alpha) * qa + alpha * qb
    result["features"][15:19] = quat / np.linalg.norm(quat)
    if time >= times[-1]:
        result["velocity"][:] = 0.
        result["qvel"][:] = 0.
    return result


def match_reference(joint_q, joint_velocity, features, *, clip: str, time: float) -> dict[str, float]:
    """Paper's weights/kernels; hinge differences replace quaternion differences.

    EE/COM use a common heading frame rather than global XY coordinates.
    Only the prescribed clip and timestamp can receive reward.
    """
    target = sample_reference(clip, time)
    dot = abs(float(np.dot(target["features"][15:19], features[15:19])))
    root_angle = 2. * np.arccos(np.clip(dot, 0., 1.))
    pose = np.exp(-2. * (np.sum(np.square(target["q"] - joint_q)) + root_angle ** 2))
    velocity = np.exp(-.1 * np.sum(np.square(target["velocity"] - joint_velocity)))
    ee = np.exp(-40. * np.sum(np.square(target["features"][:12] - features[:12])))
    com = np.exp(-10. * np.sum(np.square(target["features"][12:15] - features[12:15])))
    return {"imitation": float(.65 * pose + .1 * velocity + .15 * ee + .1 * com),
            "reference_pose": float(pose), "reference_velocity": float(velocity),
            "reference_ee": float(ee), "reference_com": float(com)}


class MotionClock:
    def __init__(self):
        self.clip = "prone"
        self.time = 0.
        self.recovering = False

    @property
    def duration(self):
        return float(reference_library()[f"{self.clip}_time"][-1])

    @property
    def phase(self):
        return min(self.time / self.duration, 1.)

    def reset(self, clip="prone", time=0., recovering=True):
        if clip not in CLIPS:
            raise ValueError(f"Unknown motion clip: {clip}")
        self.clip, self.time, self.recovering = clip, float(time), recovering

    def advance(self, dt):
        self.time = min(self.time + dt, self.duration)

    def observe_fall(self, sim):
        """Select one recovery clip on a fall; never search frames for reward."""
        state = sim.control_state()
        fallen = state["base_height"] < .6 or state["upright"] < .5
        if fallen and not self.recovering:
            forward_z = sim.data.xmat[sim.pelvis_id].reshape(3, 3)[2, 0]
            self.reset("supine" if forward_z > 0. else "prone")
        elif self.phase == 1. and state["base_height"] > .8 and state["upright"] > .9:
            self.recovering = False
