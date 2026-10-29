# Getting-up motion references

Source: Xue Bin Peng et al., DeepMimic (SIGGRAPH 2018).
Project: https://xbpeng.github.io/projects/DeepMimic/index.html
Repository: https://github.com/xbpeng/DeepMimic
Upstream directory: `data/motions/` on `master`, downloaded 2026-10-29.
Upstream MIT notice is preserved in `LICENSE.DeepMimic`.

Bundled source files (SHA-256):

- `humanoid3d_getup_facedown.txt`: `1bd1a96e9175f65eb66e10f225473b4467bd70c68e283f4006b2ab090e337c0b`
- `humanoid3d_getup_faceup.txt`: `5936c32d51dad9b0cd5e61df681079d160436cdce845c312f200d7a347c4040d`

`getup_deepmimic.npz` is the active derived asset for this 20-hinge robot.
Rebuild with `python tools/build_getup_reference.py` from the repository root.
The builder converts Y-up to Z-up, projects spherical joints onto hinge axes,
clamps joint limits, smooths projection artifacts, grounds poses, and adds a
one-second transition to this robot's standing pose. Original clip timing is
retained: total duration is about 4.03 s prone / 4.77 s supine.
Root and joint velocities are differentiated in MuJoCo's quaternion tangent
convention. The asset includes qpos/qvel for reference-state initialization,
heading-aligned hands/feet and COM positions, and root orientation.

Training uses a prescribed clip and monotonically advancing phase, not a bank
of freely selectable frames. Clip and time exist only in the training reward;
the actor and critic receive 189 state/history/contact features: root height
and quaternion/linear/angular velocity (11), 11 non-root bodies (13 each),
previous action (20), and contact pressure (15). Root-relative root position
is always zero and omitted; legacy joint q/qd, base velocity/gravity and home
blocks are not appended. Body kinematics still imposes constraints among
positions and orientations; this is not a claim of minimal state dimension.
Inference
loads no reference data and has no skill selector or motion clock. This omits
DeepMimic's phase / skill conditioning, so it is not the original method. Hidden
reference timing can assign different reward targets to the same physical state;
training quality must be validated empirically. Imitation borrows section 5.3, with
hinge-angle pose errors and translation-invariant heading-local EE/COM positions
adapted for this robot. No HumanUP rewards or regularization curriculum remain.

These are approximate **kinematic references**, not dynamically validated robot
demonstrations. The target model, PD gains, 50 Hz control, fixed exploration
scale and PPO hyperparameters differ from the paper. Reference initialization
and standing-contact early termination are adapted to the getting-up task.
The old `getup_reference.npz` is retained only as a historical asset for old runs;
new training never loads it. Old 89-input checkpoints are incompatible with the
new 189-input state/history/contact policy and should not be silently migrated.

Previous 157-input body-only, 160-input phase/skill-conditioned, and 245-input
concatenated models are incompatible too. New
checkpoints use the independent `_deepmimic_state` prefix.
