"""Retarget the bundled DeepMimic clips to our 20 hinge joints (offline only).

Run from the repository root: python tools/build_getup_reference.py
No optimization or network is needed. Source clips and license live in assets/motions.
This is an approximate morphology retarget, not dynamically validated ground truth.
"""
from pathlib import Path
import sys
import json
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_rl.sim import HumanoidSim
from robot_rl.motion import tracking_features

ASSETS = ROOT / "robot_rl/assets/motions"
C = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def rotation(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, q)
    return C @ mat.reshape(3, 3) @ C.T


def zxy(q):
    r = rotation(q)
    return np.array([np.arctan2(-r[0, 1], r[1, 1]),
                     np.arcsin(np.clip(r[2, 1], -1., 1.)),
                     np.arctan2(-r[2, 0], r[2, 2])])


def yx(q):
    r = rotation(q)
    return np.array([np.arctan2(r[0, 2], r[2, 2]),
                     np.arcsin(np.clip(-r[1, 2], -1., 1.))])


def heading_frame(sim, positions):
    r = sim.data.xmat[sim.pelvis_id].reshape(3, 3)
    yaw = np.arctan2(r[1, 0], r[0, 0])
    cs, sn = np.cos(yaw), np.sin(yaw)
    basis = np.array([[cs, -sn, 0.], [sn, cs, 0.], [0., 0., 1.]])
    return (positions - sim.data.xpos[sim.pelvis_id]) @ basis


def build():
    sim = HumanoidSim()
    output = {}
    for pose, source in [("prone", "facedown"), ("supine", "faceup")]:
        frames = np.asarray(json.loads((ASSETS / f"humanoid3d_getup_{source}.txt").read_text())["Frames"])
        time = np.r_[0., np.cumsum(frames[:-1, 0])]
        q = np.zeros((len(frames), sim.n_act))
        for k, f in enumerate(frames):
            chest = rotation(f[8:12])
            q[k, :2] = [np.arctan2(chest[1, 0], chest[0, 0]), np.arcsin(np.clip(-chest[2, 0], -1., 1.))]
            q[k, 2:5] = zxy(f[16:20])
            q[k, 5] = -f[20]
            q[k, 6:8] = yx(f[21:25])
            q[k, 8:11] = zxy(f[30:34])
            q[k, 11] = -f[34]
            q[k, 12:14] = yx(f[35:39])
            q[k, 14:16] = yx(f[25:29])
            q[k, 16] = f[29]
            q[k, 17:19] = yx(f[39:43])
            q[k, 19] = f[43]
        # Stay inside this robot's limits. Append a smooth transition to its own stand.
        q = np.clip(q, sim.q_low, sim.q_high)
        # Projection onto fewer hinge axes can introduce kinks; smooth the retarget,
        # not the policy actions. This same reference is used during training.
        padded = np.pad(q, ((4, 4), (0, 0)), mode="edge")
        q = np.stack([np.convolve(padded[:, j], np.ones(9) / 9., mode="valid") for j in range(sim.n_act)], axis=1)
        roots = np.column_stack([frames[:, 4], frames[:, 5:8] @ C.T])
        roots /= np.linalg.norm(roots, axis=1, keepdims=True)
        root_pos = frames[:, 1:4] @ C.T * 1.1
        for fraction in np.linspace(.02, 1., 50):
            blend = fraction * fraction * (3. - 2. * fraction)
            q = np.vstack([q, (1. - blend) * q[len(frames) - 1] + blend * sim.default_q])
            quat = (1. - blend) * roots[len(frames) - 1] + blend * np.array([1., 0., 0., 0.])
            roots = np.vstack([roots, quat / np.linalg.norm(quat)])
            root_pos = np.vstack([root_pos, root_pos[len(frames) - 1]])
            time = np.r_[time, time[len(frames) - 1] + fraction]
        features = []
        qposes = []
        sim.reset("stand")
        for joints, root_q, pos in zip(q, roots, root_pos):
            sim.data.qpos[sim.root_qpos_adr:sim.root_qpos_adr + 3] = pos
            sim.data.qpos[sim.root_qpos_adr + 3:sim.root_qpos_adr + 7] = root_q
            sim.data.qpos[sim.qpos_adr] = joints
            mujoco.mj_forward(sim.model, sim.data)
            sim._plant_on_ground()
            features.append(tracking_features(sim))
            qposes.append(sim.data.qpos[:sim.n_act + 7].copy())
        output[f"{pose}_time"] = time.astype(np.float32)
        output[f"{pose}_q"] = q.astype(np.float32)
        output[f"{pose}_velocity"] = np.gradient(q, time, axis=0).astype(np.float32)
        output[f"{pose}_features"] = np.asarray(features, dtype=np.float32)
        output[f"{pose}_qpos"] = np.asarray(qposes, dtype=np.float32)
        # Differentiate free-root quaternions using MuJoCo's tangent convention.
        qvel = []
        full_a, full_b = sim.data.qpos.copy(), sim.data.qpos.copy()
        for k in range(len(qposes)):
            left, right = max(0, k - 1), min(len(qposes) - 1, k + 1)
            full_a[:sim.n_act + 7] = qposes[left]
            full_b[:sim.n_act + 7] = qposes[right]
            velocity = np.empty(sim.model.nv)
            mujoco.mj_differentiatePos(sim.model, velocity, time[right] - time[left], full_a, full_b)
            qvel.append(np.r_[velocity[sim.root_dof_adr:sim.root_dof_adr + 6], velocity[sim.dof_adr]])
        qvel = np.asarray(qvel, dtype=np.float32)
        qvel[-1] = 0.
        output[f"{pose}_qvel"] = qvel
        output[f"{pose}_velocity"] = qvel[:, 6:]
        print(pose, "frames", len(q), "seconds", time[-1], "COM height range", np.asarray(features)[:, 14].min(), np.asarray(features)[:, 14].max())
    np.savez_compressed(ASSETS / "getup_deepmimic.npz", **output)


if __name__ == "__main__":
    build()
