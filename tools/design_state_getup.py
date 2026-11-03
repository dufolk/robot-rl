"""Forward-only physical event control; no time or phase inputs."""
from pathlib import Path
import json
import numpy as np
from tools.build_manual_getup import *


def rollout(parameters=None):
    s=HumanoidSim(physics_dt=.001,action_mapping='joint_limits')
    out=Path('artifacts/manual_getup')
    v=np.load(out/'manual_parameters.npz')['parameters']
    table=latent_table(s,v[:49].reshape(7,7))
    route=np.load(out/'transition.npz')['parameters']
    qs=[joint_pose(q,s) for q in route[:21].reshape(3,7)]
    s.reset('supine');rows=[];qposes=[];stages=[]
    for k in range(1000):
        h=s.data.xpos[s.pelvis_id,2];tilt=get_tilt(s);q=s.data.qpos[s.qpos_adr]
        if h>.72:
            target=s.default_q;stage=4
        elif h>.36 and q[1]<.4 and tilt>-.5:
            target=qs[2];stage=3
        elif h>.27 and (q[5]>2.05 or q[1]<.2):
            target=qs[1];stage=2
        elif h>.275 and tilt>-.27 and tilt<-.15:
            target=qs[0];stage=1
        else:
            target=s.action_targets(command(s,table,v[49],v[50]));stage=0
        delta=target-s.default_q;a=np.clip(delta/np.where(delta>=0,s.action_scale_high,s.action_scale_low),-.964,.964)
        s.apply_action(a);fast_step(s,s.q_target);rows.append(metrics(s));qposes.append(s.data.qpos[:27].copy());stages.append(stage)
    np.savez(out/'state_replay.npz',metrics=rows,qpos=qposes,stages=stages)
    print(np.array(rows)[::50],np.array(rows)[-150:].mean(0));print(np.array(stages)[::25])


if __name__=='__main__':rollout()
