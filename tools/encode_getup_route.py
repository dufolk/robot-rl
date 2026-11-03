"""Encode a forward-validated route as physical-state receptive-field neurons.

Hidden layer 1 computes absolute distances from hand-selected physical states;
hidden layer 2 builds compact receptive fields. The output head is calculated
with a linear system (no gradients, optimizer, backprop or pretrained weights).
No time, phase, reference pose or skill identifier is an input.
"""
import argparse
import io
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np
import torch
from robot_rl.humanup import ObservationHistory
from robot_rl.checkpoint import InferencePolicy
from tools.build_manual_getup import latent_table, command, save_model, explicit_weights
from tools.design_getup_forward import HumanoidSim, fast_step, metrics, joint_pose

# Current physical pelvis height / heading-local quaternion tilt / last knee
# target, plus latest real hip angular position from the compact history.
FEATURES = np.array([0,3,159])
SCALES = np.array([1.,1.5,.6])
N_CENTERS=42


def collect(out):
    sim=HumanoidSim(physics_dt=.001,action_mapping='joint_limits')
    v=np.load(out/'route_initial_parameters.npz')['parameters']
    table=latent_table(sim,v[:49].reshape(7,7))
    route=np.load(out/'transition.npz')
    sim.reset('supine');hist=ObservationHistory();hist.reset(sim)
    observations=[];actions=[];rows=[];qposes=[]
    def step(target):
        observations.append(hist.observation(sim))
        delta=target-sim.default_q
        a=np.clip(delta/np.where(delta>=0,sim.action_scale_high,sim.action_scale_low),-.964,.964)
        actions.append(a.copy())
        sim.apply_action(a);fast_step(sim,sim.action_targets(a));sim._read_contacts();hist.advance(sim)
        rows.append(metrics(sim));qposes.append(sim.data.qpos[:27].copy())
    for k in range(160):
        step(sim.action_targets(command(sim,table,v[49],v[50])))
    prev=route['initial_target'].copy()
    pars=route['parameters']
    qs=[joint_pose(q,sim) for q in pars[:21].reshape(3,7)]+[sim.default_q]
    for q,dur in zip(qs,[*pars[21:],1.]):
        n=max(1,round(dur*50))
        for k in range(n):
            a=(k+1)/n;a=a*a*(3-2*a);step(prev*(1-a)+q*a)
        prev=q.copy()
    for k in range(250):step(sim.default_q)
    x,y=np.array(observations),np.array(actions)
    np.savez(out/'forward_route.npz',observations=x,actions=y,metrics=rows,qpos=qposes)
    return x,y


def construct(x,y,width=2.,ridge=1e-3):
    points=x[:,FEATURES]*SCALES
    # Preserve dense moving sections rather than spending neurons on stillness.
    length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    indices=np.unique(np.searchsorted(length,np.linspace(0,length[-1],N_CENTERS)))
    centers=points[indices]
    n=len(centers)
    distance=np.sum(abs(centers[:,None]-centers[None,:]),axis=-1)
    distance+=np.eye(n)*1000
    radii=np.maximum(.08,np.partition(distance,2,axis=1)[:,2]*width)
    state=explicit_weights(np.zeros((7,20)))
    for key in state:
        if key!='log_std':state[key].zero_()
    w1=state['mlp_extractor.policy_net.0.weight'];b1=state['mlp_extractor.policy_net.0.bias']
    w2=state['mlp_extractor.policy_net.2.weight'];b2=state['mlp_extractor.policy_net.2.bias']
    for i,center in enumerate(centers):
        for d,feature in enumerate(FEATURES):
            for sign,offset in [(1,0),(-1,1)]:
                row=i*6+d*2+offset
                w1[row,int(feature)]=float(sign*SCALES[d])
                b1[row]=float(-sign*center[d])
                w2[i,row]=-1/float(radii[i])
        b2[i]=1.
    # Output coefficients from an ordinary linear solve over forward activations.
    # Ridge stabilizes overlapping receptive fields; no policy gradient exists.
    phi=np.maximum(0,1-np.sum(abs(points[:,None]-centers[None,:]),axis=-1)/radii)
    design=np.column_stack([phi,np.ones(len(phi))])
    latent=2*np.arctanh(np.arctanh(np.clip(y,-.95,.95))/2)
    regularizer=np.eye(n+1)*ridge
    coef=np.linalg.solve(design.T@design+regularizer,design.T@latent)
    state['action_net.weight'][:,:n]=torch.tensor(coef[:n].T,dtype=torch.float32)
    state['action_net.bias'][:]=torch.tensor(coef[-1],dtype=torch.float32)
    return state,float(np.mean((design@coef-latent)**2))


def attach_recovery(state, x, y, threshold=.28, radius=.7, ridge=.001):
    """Keep the measured low-state recovery exactly, fit only its residual."""
    out=Path('artifacts/manual_getup')
    sim=HumanoidSim(physics_dt=.001,action_mapping='joint_limits')
    base=np.load(out/'route_initial_parameters.npz')['parameters']
    original=explicit_weights(latent_table(sim,base[:49].reshape(7,7)),base[49],base[50])
    # First 8 and second 40 neurons preserve the analytical recovery network.
    # Subtract its latent output with physical-state compact residual fields.
    w1=original['mlp_extractor.policy_net.0.weight'];b1=original['mlp_extractor.policy_net.0.bias']
    w2=original['mlp_extractor.policy_net.2.weight'];b2=original['mlp_extractor.policy_net.2.bias']
    idx=np.arange(160,min(len(x),350),2)
    points=x[:,[0,155,159]]*np.array([1,1.,1.])
    centers=points[idx]
    n=min(40,len(centers));indices=np.linspace(0,len(centers)-1,n).astype(int);centers=centers[indices]
    distances=np.sum(abs(centers[:,None]-centers[None,:]),axis=-1)+np.eye(n)*1000
    radii=np.maximum(.12,np.partition(distances,2,axis=1)[:,2]*radius)
    features=[0,155,159];scales=[1,1,1]
    w1[248,0]=-1.;b1[248]=threshold
    for i,c in enumerate(centers):
        for d,f in enumerate(features):
            for sign,k in [(1,0),(-1,1)]:
                row=8+i*6+d*2+k;w1[row,f]=sign*scales[d];b1[row]=-sign*float(c[d]);w2[40+i,row]=-1/float(radii[i])
        b2[40+i]=1
        w2[40+i,248]=-200.
    with torch.inference_mode():
        z=torch.relu(torch.tensor(x)@w1.T+b1);h=torch.relu(z@w2.T+b2)
        base_raw=h@original['action_net.weight'].T
    phi=h[:,40:40+n].numpy()
    latent=2*np.arctanh(np.arctanh(np.clip(y,-.964,.964))/2)
    coef=np.linalg.solve(phi.T@phi+np.eye(n)*ridge,phi.T@(latent-base_raw.numpy()))
    original['action_net.weight'][:,40:40+n]=torch.tensor(coef.T,dtype=torch.float32)
    return original


def save(path,state):
    # Metadata comes from a newly constructed schema, never a learned checkpoint.
    save_model(path,np.zeros((7,20)),.48,14)
    with zipfile.ZipFile(path) as ar:metadata=ar.read('data')
    payload=io.BytesIO();torch.save(state,payload)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as ar:
        ar.writestr('data',metadata);ar.writestr('policy.pth',payload.getvalue())


def replay(path,pose,steps=1000):
    policy=InferencePolicy(path);s=HumanoidSim(physics_dt=.001,action_mapping='joint_limits');s.reset(pose)
    hist=ObservationHistory();hist.reset(s);rows=[];qposes=[];observations=[];actions=[]
    for k in range(steps):
        obs=hist.observation(s);a=policy.predict(obs)[0];s.apply_action(a);fast_step(s,s.q_target);s._read_contacts();hist.advance(s)
        rows.append(metrics(s));qposes.append(s.data.qpos[:27].copy());observations.append(obs);actions.append(a)
    rows=np.array(rows)
    quality=np.minimum(np.clip(rows[:,0]/.94,0,1),np.clip(rows[:,1]/1.5,0,1))*np.maximum(0,np.minimum(rows[:,2],rows[:,3]))
    score=5*quality[-150:].mean()+2*quality.max()+quality.mean()
    return score,rows,np.array(qposes),np.array(observations),np.array(actions)


def main():
    torch.set_grad_enabled(False);torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='artifacts/manual_getup');args=p.parse_args()
    out=Path(args.output);x,y=collect(out)
    best=-1;results=[]
    for width in [.7,1.,1.5,2.,3.,5.,8.]:
        for ridge in [.001,.03]:
            state,error=construct(x,y,width,ridge);path=out/'candidate_route.zip';save(path,state)
            score,rows,qp,obs,acts=replay(path,'supine',steps=650)
            rec=dict(width=width,ridge=ridge,score=score,error=error,tail=rows[-150:].mean(0).tolist());results.append(rec);print(json.dumps(rec),flush=True)
            if score>best:
                best=score;save(out/'getup_route.zip',state);np.savez(out/'encoded_replay.npz',metrics=rows,qpos=qp,observations=obs,actions=acts)
    (out/'encoding_report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
