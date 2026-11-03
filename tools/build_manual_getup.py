"""Construct every MLP tensor explicitly; optional forward-only CEM search.

No learned checkpoint is read, no optimizer is created, and autograd is disabled.
First hidden layer measures torso tilt using piecewise linear ramps. The second
layer selects positive/negative joint commands and withdraws recovery commands
as the pelvis reaches standing height. The output is a bounded PD target.
"""
import argparse
import io
import json
from pathlib import Path
import zipfile

import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3.common.save_util import data_to_json

from robot_rl.humanup_policy import RecoveryPolicy
from robot_rl.humanup import MODEL_CONTRACT, ObservationHistory
from robot_rl.checkpoint import InferencePolicy
from robot_rl.motion import heading_basis
from robot_rl.sim import HumanoidSim
from tools.design_getup_forward import joint_pose, fast_step, metrics

KNOTS = np.array([-1., -.7, -.4, 0., .4, .7, 1.])
POSES = np.array([
    [.8, -1.5, 2.3, -.7, -.5, .2, .2],
    [.8, -1.5, 2.3, -.7, -.5, .2, .2],
    [.8, -1., 2.3, -.8, .8, .1, .2],
    [.2, -1.2, 2.2, -.8, -.35, .55, .25],
    [-.3, -1., 2.2, -.7, -1.3, .7, .25],
    [0., .2, .6, .2, -.2, 1.5, .4],
    [0., .2, .6, .2, -.2, 1.5, .4],
])


def latent_table(sim, poses):
    q = np.stack([joint_pose(v, sim) for v in poses])
    delta = q - sim.default_q
    action = delta / np.where(delta >= 0, sim.action_scale_high, sim.action_scale_low)
    action = np.clip(action, -.95, .95)
    return 2 * np.arctanh(np.arctanh(action) / 2.)


def explicit_weights(table, cutoff=.48, withdrawal=14.):
    """All unused connections are explicitly zero, keeping 334121 parameters."""
    state = {}
    for name, shape in [('mlp_extractor.policy_net.0.weight',(256,384)),
                        ('mlp_extractor.policy_net.0.bias',(256,)),
                        ('mlp_extractor.policy_net.2.weight',(256,256)),
                        ('mlp_extractor.policy_net.2.bias',(256,)),
                        ('action_net.weight',(20,256)), ('action_net.bias',(20,)),
                        ('mlp_extractor.value_net.0.weight',(256,384)),
                        ('mlp_extractor.value_net.0.bias',(256,)),
                        ('mlp_extractor.value_net.2.weight',(256,256)),
                        ('mlp_extractor.value_net.2.bias',(256,)),
                        ('value_net.weight',(1,256)), ('value_net.bias',(1,)),
                        ('log_std',(20,))]:
        state[name] = torch.zeros(shape)
    w1, b1 = state['mlp_extractor.policy_net.0.weight'],state['mlp_extractor.policy_net.0.bias']
    # q_y is index 3 of heading-local pelvis quaternion (w,x,y,z).
    for k, knot in enumerate(KNOTS):
        w1[k,3] = 1
        b1[k] = -float(knot)
    # Standing-height command withdrawal; physical height only.
    w1[7,0] = 1
    b1[7] = -float(cutoff)
    slopes = np.diff(table,axis=0) / np.diff(KNOTS)[:,None]
    coefficients = np.vstack([slopes[0],np.diff(slopes,axis=0),-slopes[-1]])
    w2, b2 = state['mlp_extractor.policy_net.2.weight'],state['mlp_extractor.policy_net.2.bias']
    out = state['action_net.weight']
    for j in range(20):
        for sign, row in [(1,2*j),(-1,2*j+1)]:
            w2[row,:7] = torch.tensor(sign*coefficients[:,j],dtype=torch.float32)
            w2[row,7] = -withdrawal
            b2[row] = float(sign*table[0,j])
            out[j,row] = sign
    state['log_std'].fill_(-5)
    return state


def save_model(path, table, cutoff, withdrawal):
    weights = explicit_weights(table,cutoff,withdrawal)
    metadata = dict(policy_class=RecoveryPolicy,
                    policy_kwargs=dict(net_arch=dict(pi=[256,256],vf=[256,256]),activation_fn=torch.nn.ReLU),
                    observation_space=gym.spaces.Box(-np.inf,np.inf,(384,),dtype=np.float32),
                    action_space=gym.spaces.Box(-1,1,(20,),dtype=np.float32),
                    humanup_contract=MODEL_CONTRACT, construction='explicit-forward-only-v1',
                    num_timesteps=0, backpropagation_steps=0,
                    humanup_stage='manual', getup_reward=None)
    payload = io.BytesIO()
    torch.save(weights,payload)
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('data',data_to_json(metadata))
        archive.writestr('policy.pth',payload.getvalue())
        archive.writestr('_stable_baselines3_version','2.7.1')
    return sum(v.numel() for v in weights.values())


def get_tilt(sim):
    quat=np.empty(4)
    mujoco.mju_mat2Quat(quat,(heading_basis(sim).T @ sim.data.xmat[sim.pelvis_id].reshape(3,3)).ravel())
    if quat[0] < 0: quat *= -1
    return quat[2]


def command(sim,table,cutoff,withdrawal):
    tilt=get_tilt(sim)
    raw=np.array([np.interp(tilt,KNOTS,table[:,j]) for j in range(20)])
    suppress=max(0.,sim.data.xpos[sim.pelvis_id,2]-cutoff)*withdrawal
    raw=np.sign(raw)*np.maximum(0.,np.abs(raw)-suppress)
    return np.tanh(2*np.tanh(raw/2))


def evaluate(sim,table,cutoff,withdrawal,pose='supine',steps=600):
    sim.reset(pose)
    rows=[]
    for k in range(steps):
        action=command(sim,table,cutoff,withdrawal)
        fast_step(sim,sim.action_targets(action))
        rows.append(metrics(sim))
    rows=np.array(rows)
    clearance=np.minimum(np.clip(rows[:,0]/.94,0,1),np.clip(rows[:,1]/1.5,0,1))
    quality=clearance*np.maximum(0,np.minimum(rows[:,2],rows[:,3]))
    tail=rows[-150:]
    score=5*np.mean(quality[-150:])+2*np.max(quality)+np.mean(quality)-.05*np.mean(np.minimum(tail[:,4]+tail[:,5],10))
    return score,rows


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iterations',type=int,default=0)
    p.add_argument('--population',type=int,default=48)
    p.add_argument('--output',default='artifacts/manual_getup')
    p.add_argument('--seed',type=int,default=7)
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    sim=HumanoidSim(physics_dt=.001,action_mapping='joint_limits')
    rng=np.random.default_rng(args.seed)
    mean=np.r_[POSES.ravel(),.48,14.]
    lower=np.r_[np.tile([-.5,-1.6,0,-.8,-2.4,0,.08],7),.35,4]
    upper=np.r_[np.tile([.8,1.1,2.3,.8,1.2,2.3,.65],7),.85,30]
    std=np.r_[np.tile([.25,.5,.5,.3,.8,.6,.2],7),.1,4.]
    record=out/'manual_parameters.npz'
    if record.exists():
        mean=np.load(record)['parameters'].copy()
    best=mean.copy()
    def assess(v):
        return evaluate(sim,latent_table(sim,v[:49].reshape(7,7)),v[49],v[50])[0]
    best_score=assess(best)
    for it in range(args.iterations):
        candidates=np.clip(mean+rng.normal(size=(args.population,len(mean)))*std,lower,upper)
        candidates[0]=best
        scores=np.array([assess(v) for v in candidates])
        order=np.argsort(scores)[::-1]
        elites=candidates[order[:max(5,args.population//6)]]
        mean=.15*mean+.85*elites.mean(0)
        std=np.maximum(.025,.4*std+.6*elites.std(0))
        if scores[order[0]]>best_score:
            best_score=float(scores[order[0]]);best=candidates[order[0]].copy()
        np.savez(record,parameters=best,score=best_score)
        save_model(out/'getup_manual.zip',latent_table(sim,best[:49].reshape(7,7)),best[49],best[50])
        print(json.dumps(dict(iteration=it,best=best_score)),flush=True)
    table=latent_table(sim,best[:49].reshape(7,7))
    count=save_model(out/'getup_manual.zip',table,best[49],best[50])
    policy=InferencePolicy(out/'getup_manual.zip')
    results={}
    for pose in ['stand','supine','prone']:
        score,rows=evaluate(sim,table,best[49],best[50],pose=pose,steps=1000)
        # Verify the exported network equals the forward search controller.
        history=ObservationHistory();history.reset(sim)
        np.testing.assert_allclose(policy.predict(history.observation(sim))[0],command(sim,table,best[49],best[50]),atol=2e-6)
        np.save(out/f'{pose}_metrics.npy',rows)
        recovered=(rows[:,0]>=.8)&(rows[:,2]>=.9)&(rows[:,3]>=.9)&(rows[:,4]<.25)&(rows[:,5]<.6)
        results[pose]=dict(score=score,tail_mean=rows[-150:].mean(0).tolist(),geometric_stable_seconds=float(np.sum(recovered[-250:])*.02))
    results['parameters']=count
    results['backpropagation_steps']=0
    results['architecture']=[384,256,256,20]
    (out/'report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
