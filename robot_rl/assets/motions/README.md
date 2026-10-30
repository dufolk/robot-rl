# Archived DeepMimic assets

No active course or inference policy loads these files. The getup course now
uses HumanUP discovery, then optional refinement from this robot's own verified
rollouts exported by `python -m robot_rl.export_humanup`.

Historical source: Xue Bin Peng et al., DeepMimic (SIGGRAPH 2018).
Project: https://xbpeng.github.io/projects/DeepMimic/index.html
Repository: https://github.com/xbpeng/DeepMimic
Downloaded from `data/motions/` on master, 2026-10-29.
The upstream MIT notice is preserved in `LICENSE.DeepMimic`.

Source SHA-256:

- `humanoid3d_getup_facedown.txt`: `1bd1a96e9175f65eb66e10f225473b4467bd70c68e283f4006b2ab090e337c0b`
- `humanoid3d_getup_faceup.txt`: `5936c32d51dad9b0cd5e61df681079d160436cdce845c312f200d7a347c4040d`

`getup_reference.npz` and `getup_deepmimic.npz` are old approximate kinematic
retargets for the 20-hinge model, not validated robot demonstrations. The offline
`tools/build_getup_reference.py` builder is retained for historical reproducibility.
These assets cannot be substituted for a HumanUP Stage I discovery trajectory.
