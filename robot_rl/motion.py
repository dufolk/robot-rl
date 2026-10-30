"""Shared physical observations; no task clock or reference data is read.

Legacy body/fused layouts remain for viewing old checkpoints. The active policy
layout is physical_observation; HumanUP appends real observation history.
"""
import numpy as np
import mujoco


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
